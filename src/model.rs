use crate::block::MiniMindBlock;
use crate::cache::KvCache;
use crate::config::MiniMindConfig;
use crate::rms_norm::RmsNorm;
use crate::rope::RotaryEmbedding;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, VarBuilder};

#[derive(Debug, Clone)]
pub struct MiniMindModel {
    pub embed_tokens: Embedding,
    pub layers: Vec<MiniMindBlock>,
    pub norm: RmsNorm,
    pub rope: RotaryEmbedding,
    pub config: MiniMindConfig,
}

impl MiniMindModel {
    pub fn load(config: &MiniMindConfig, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let layers_vb = vb.pp("layers");
        for i in 0..config.num_hidden_layers {
            let layer = MiniMindBlock::load(i, config, layers_vb.pp(i.to_string()))?;
            layers.push(layer);
        }

        let norm = RmsNorm::load(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;

        let rope = RotaryEmbedding::new(
            config.head_dim(),
            config.max_position_embeddings,
            config.rope_theta,
            config.inference_rope_scaling,
            vb.device(),
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rope,
            config: config.clone(),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        mut cache: Option<&mut KvCache>,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (_bsz, seq_len) = input_ids.dims2()?;
        let start_pos = cache.as_ref().map(|c| c.current_seq_len()).unwrap_or(0);

        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        let (cos, sin) = self.rope.get_embeddings(start_pos, seq_len)?;

        let mut total_aux_loss: Option<Tensor> = None;

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_cache = cache.as_mut().map(|c| &mut c.layers[i]);
            let (next_hidden, aux_loss) =
                layer.forward(&hidden_states, &cos, &sin, layer_cache, attention_mask)?;
            hidden_states = next_hidden;

            if let Some(aux) = aux_loss {
                total_aux_loss = match total_aux_loss {
                    Some(prev) => Some((prev + aux)?),
                    None => Some(aux),
                };
            }
        }

        let hidden_states = self.norm.forward(&hidden_states)?;
        Ok((hidden_states, total_aux_loss))
    }
}

#[derive(Debug, Clone)]
pub struct MiniMindForCausalLM {
    pub model: MiniMindModel,
    pub lm_head: Linear,
    pub config: MiniMindConfig,
}

pub struct CausalLmOutput {
    pub logits: Tensor,
    pub loss: Option<Tensor>,
    pub aux_loss: Option<Tensor>,
}

impl MiniMindForCausalLM {
    pub fn load(config: &MiniMindConfig, vb: VarBuilder) -> Result<Self> {
        let model = MiniMindModel::load(config, vb.pp("model"))?;

        let lm_head = if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            model,
            lm_head,
            config: config.clone(),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        cache: Option<&mut KvCache>,
        attention_mask: Option<&Tensor>,
        labels: Option<&Tensor>,
    ) -> Result<CausalLmOutput> {
        let (hidden_states, aux_loss) = self.model.forward(input_ids, cache, attention_mask)?;
        let logits = self.lm_head.forward(&hidden_states)?;

        let loss = if let Some(labels) = labels {
            Some(self.compute_loss(&logits, labels)?)
        } else {
            None
        };

        Ok(CausalLmOutput {
            logits,
            loss,
            aux_loss,
        })
    }

    pub fn compute_loss(&self, logits: &Tensor, labels: &Tensor) -> Result<Tensor> {
        let (_bsz, seq_len, vocab_size) = logits.dims3()?;
        if seq_len <= 1 {
            return Tensor::new(0.0f32, logits.device());
        }

        let shift_logits = logits.narrow(1, 0, seq_len - 1)?.contiguous()?;
        let shift_labels = labels.narrow(1, 1, seq_len - 1)?.contiguous()?;

        let flat_logits = shift_logits.reshape(((seq_len - 1) * _bsz, vocab_size))?;
        let flat_labels = shift_labels.reshape(((seq_len - 1) * _bsz,))?;

        let log_probs = candle_nn::ops::log_softmax(&flat_logits, candle_core::D::Minus1)?;
        let labels_vec: Vec<i64> = flat_labels.to_dtype(DType::I64)?.to_vec1()?;

        let mut total_loss = 0.0f32;
        let mut count = 0usize;

        let log_probs_vec = log_probs.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        for (i, &label) in labels_vec.iter().enumerate() {
            if label >= 0 && (label as usize) < vocab_size {
                let target_logp = log_probs_vec[i][label as usize];
                total_loss -= target_logp;
                count += 1;
            }
        }

        if count > 0 {
            Tensor::new(total_loss / count as f32, logits.device())
        } else {
            Tensor::new(0.0f32, logits.device())
        }
    }

    pub fn device(&self) -> &Device {
        self.model.embed_tokens.embeddings().device()
    }
}
