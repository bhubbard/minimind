use candle_core::{Device, Result, Tensor};

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    pub freqs_cos: Tensor,
    pub freqs_sin: Tensor,
}

impl RotaryEmbedding {
    pub fn new(
        dim: usize,
        end: usize,
        rope_base: f32,
        inference_rope_scaling: bool,
        device: &Device,
    ) -> Result<Self> {
        let half_dim = dim / 2;
        let mut freqs: Vec<f32> = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            let exponent = (2 * i) as f32 / dim as f32;
            freqs.push(1.0 / rope_base.powf(exponent));
        }

        let mut attn_factor = 1.0f32;
        if inference_rope_scaling {
            let orig_max = 2048.0f64;
            let factor = 16.0f64;
            let beta_fast = 32.0f64;
            let beta_slow = 1.0f64;
            attn_factor = 1.0;

            if (end as f64) / orig_max > 1.0 {
                let inv_dim = |b: f64| -> f64 {
                    (dim as f64 * (orig_max / (b * 2.0 * std::f64::consts::PI)).ln())
                        / (2.0 * (rope_base as f64).ln())
                };

                let low = inv_dim(beta_fast).floor().max(0.0) as usize;
                let high = (inv_dim(beta_slow).ceil() as usize).min(half_dim.saturating_sub(1));
                let diff = (high as f64 - low as f64).max(0.001);

                for i in 0..half_dim {
                    let ramp = if i < low {
                        0.0
                    } else if i > high {
                        1.0
                    } else {
                        ((i as f64 - low as f64) / diff).clamp(0.0, 1.0)
                    };
                    freqs[i] = (freqs[i] as f64 * (1.0 - ramp + ramp / factor)) as f32;
                }
            }
        }

        let mut cos_table = Vec::with_capacity(end * dim);
        let mut sin_table = Vec::with_capacity(end * dim);

        for t in 0..end {
            let mut cos_row_half = Vec::with_capacity(half_dim);
            let mut sin_row_half = Vec::with_capacity(half_dim);
            for i in 0..half_dim {
                let val = (t as f32) * freqs[i];
                cos_row_half.push(val.cos() * attn_factor);
                sin_row_half.push(val.sin() * attn_factor);
            }
            // torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
            cos_table.extend_from_slice(&cos_row_half);
            cos_table.extend_from_slice(&cos_row_half);
            sin_table.extend_from_slice(&sin_row_half);
            sin_table.extend_from_slice(&sin_row_half);
        }

        let freqs_cos = Tensor::from_vec(cos_table, (end, dim), device)?;
        let freqs_sin = Tensor::from_vec(sin_table, (end, dim), device)?;

        Ok(Self {
            freqs_cos,
            freqs_sin,
        })
    }

    pub fn get_embeddings(&self, start_pos: usize, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let cos = self.freqs_cos.narrow(0, start_pos, seq_len)?;
        let sin = self.freqs_sin.narrow(0, start_pos, seq_len)?;
        Ok((cos, sin))
    }
}

pub fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last_dim = x.dim(candle_core::D::Minus1)?;
    let half_dim = last_dim / 2;
    let x1 = x.narrow(candle_core::D::Minus1, 0, half_dim)?;
    let x2 = x.narrow(candle_core::D::Minus1, half_dim, half_dim)?;
    let neg_x2 = x2.neg()?;
    Tensor::cat(&[&neg_x2, &x1], candle_core::D::Minus1)
}

pub fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // q and k: [batch_size, seq_len, num_heads, head_dim]
    // cos and sin: [seq_len, head_dim] -> unsqueeze(0) -> [1, seq_len, 1, head_dim]
    let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

    let q_embed = (q.broadcast_mul(&cos)? + rotate_half(q)?.broadcast_mul(&sin)?)?;
    let k_embed = (k.broadcast_mul(&cos)? + rotate_half(k)?.broadcast_mul(&sin)?)?;

    Ok((q_embed, k_embed))
}
