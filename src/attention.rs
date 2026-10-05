use crate::cache::LayerCache;
use crate::config::MiniMindConfig;
use crate::rms_norm::RmsNorm;
use crate::rope::apply_rotary_pos_emb;
use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

#[derive(Debug, Clone)]
pub struct Attention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub n_rep: usize,
    pub hidden_size: usize,
}

impl Attention {
    pub fn load(config: &MiniMindConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = config.head_dim();
        let num_attention_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads;
        let hidden_size = config.hidden_size;
        let n_rep = config.num_rep();

        let q_proj = linear_no_bias(hidden_size, num_attention_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_size, num_key_value_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_size, num_key_value_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_attention_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        let q_norm = RmsNorm::load(head_dim, config.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::load(head_dim, config.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            n_rep,
            hidden_size,
        })
    }

    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        if self.n_rep == 1 {
            return Ok(x.clone());
        }
        let (bsz, seq_len, num_kv_heads, head_dim) = x.dims4()?;
        let expanded = x
            .unsqueeze(3)?
            .broadcast_as((bsz, seq_len, num_kv_heads, self.n_rep, head_dim))?;
        expanded.contiguous()?.reshape((bsz, seq_len, num_kv_heads * self.n_rep, head_dim))
    }

    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cache: Option<&mut LayerCache>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (bsz, seq_len, _) = x.dims3()?;

        let xq = self.q_proj.forward(x)?;
        let xk = self.k_proj.forward(x)?;
        let xv = self.v_proj.forward(x)?;

        let xq = xq.reshape((bsz, seq_len, self.num_attention_heads, self.head_dim))?;
        let xk = xk.reshape((bsz, seq_len, self.num_key_value_heads, self.head_dim))?;
        let xv = xv.reshape((bsz, seq_len, self.num_key_value_heads, self.head_dim))?;

        let xq = self.q_norm.forward(&xq)?;
        let xk = self.k_norm.forward(&xk)?;

        let (xq, xk) = apply_rotary_pos_emb(&xq, &xk, cos, sin)?;

        let (xk, xv) = match cache {
            Some(c) => c.update(xk, xv)?,
            None => (xk, xv),
        };

        let xk = self.repeat_kv(&xk)?;
        let xv = self.repeat_kv(&xv)?;

        let q = xq.transpose(1, 2)?.contiguous()?;
        let k = xk.transpose(1, 2)?.contiguous()?;
        let v = xv.transpose(1, 2)?.contiguous()?;

        let total_kv_len = k.dim(2)?;

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k_t)? * scale)?;

        let scores = if seq_len > 1 {
            let mask = self.make_causal_mask(seq_len, total_kv_len, x.device(), scores.dtype())?;
            scores.broadcast_add(&mask)?
        } else {
            scores
        };

        let scores = if let Some(att_mask) = attention_mask {
            let mask_4d = att_mask.unsqueeze(1)?.unsqueeze(2)?;
            let zero = Tensor::zeros_like(&mask_4d)?;
            let neg_inf = Tensor::full(-1e9f32, mask_4d.shape(), x.device())?.to_dtype(scores.dtype())?;
            let bias = mask_4d.where_cond(&zero, &neg_inf)?;
            scores.broadcast_add(&bias)?
        } else {
            scores
        };

        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let output = weights.matmul(&v)?;

        let output = output.transpose(1, 2)?.contiguous()?;
        let output = output.reshape((bsz, seq_len, self.num_attention_heads * self.head_dim))?;

        self.o_proj.forward(&output)
    }

    fn make_causal_mask(
        &self,
        seq_len: usize,
        total_len: usize,
        device: &candle_core::Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let mut mask_vec = vec![0.0f32; seq_len * total_len];
        let offset = total_len.saturating_sub(seq_len);
        for i in 0..seq_len {
            for j in 0..total_len {
                if j > i + offset {
                    mask_vec[i * total_len + j] = -1e9f32;
                }
            }
        }
        Tensor::from_vec(mask_vec, (1, 1, seq_len, total_len), device)?.to_dtype(dtype)
    }
}
