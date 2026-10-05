use crate::attention::Attention;
use crate::cache::LayerCache;
use crate::config::MiniMindConfig;
use crate::mlp::{FeedForward, Mlp, MoeFeedForward};
use crate::rms_norm::RmsNorm;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

#[derive(Debug, Clone)]
pub struct MiniMindBlock {
    pub self_attn: Attention,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
    pub mlp: Mlp,
}

impl MiniMindBlock {
    pub fn load(layer_id: usize, config: &MiniMindConfig, vb: VarBuilder) -> Result<Self> {
        let _ = layer_id;
        let self_attn = Attention::load(config, vb.pp("self_attn"))?;
        let input_layernorm = RmsNorm::load(config.hidden_size, config.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            RmsNorm::load(config.hidden_size, config.rms_norm_eps, vb.pp("post_attention_layernorm"))?;

        let mlp = if config.use_moe {
            Mlp::Moe(MoeFeedForward::load(config, vb.pp("mlp"))?)
        } else {
            Mlp::Dense(FeedForward::load(
                config.hidden_size,
                config.intermediate_size(),
                vb.pp("mlp"),
            )?)
        };

        Ok(Self {
            self_attn,
            input_layernorm,
            post_attention_layernorm,
            mlp,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cache: Option<&mut LayerCache>,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        // Pre-norm RMSNorm -> Attention -> Residual
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_out = self.self_attn.forward(&normed, cos, sin, cache, attention_mask)?;
        let hidden_states = (hidden_states + attn_out)?;

        // Pre-norm RMSNorm -> MLP -> Residual
        let normed = self.post_attention_layernorm.forward(&hidden_states)?;
        let (mlp_out, aux_loss) = self.mlp.forward(&normed)?;
        let hidden_states = (hidden_states + mlp_out)?;

        Ok((hidden_states, aux_loss))
    }
}
