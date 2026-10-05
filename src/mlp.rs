use crate::config::MiniMindConfig;
use candle_core::{Module, Result, Tensor};
use candle_nn::{linear_no_bias, ops::silu, Linear, VarBuilder};

#[derive(Debug, Clone)]
pub struct FeedForward {
    pub gate_proj: Linear,
    pub down_proj: Linear,
    pub up_proj: Linear,
}

impl FeedForward {
    pub fn load(
        hidden_size: usize,
        intermediate_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        Ok(Self {
            gate_proj,
            down_proj,
            up_proj,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = silu(&gate)?;
        let intermediate = activated.mul(&up)?;
        self.down_proj.forward(&intermediate)
    }
}

#[derive(Debug, Clone)]
pub struct MoeFeedForward {
    pub gate: Linear,
    pub experts: Vec<FeedForward>,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub norm_topk_prob: bool,
    pub router_aux_loss_coef: f64,
}

impl MoeFeedForward {
    pub fn load(config: &MiniMindConfig, vb: VarBuilder) -> Result<Self> {
        let gate = linear_no_bias(config.hidden_size, config.num_experts, vb.pp("gate"))?;
        let mut experts = Vec::with_capacity(config.num_experts);
        let experts_vb = vb.pp("experts");
        for i in 0..config.num_experts {
            let expert = FeedForward::load(
                config.hidden_size,
                config.moe_intermediate_size(),
                experts_vb.pp(i.to_string()),
            )?;
            experts.push(expert);
        }
        Ok(Self {
            gate,
            experts,
            num_experts: config.num_experts,
            num_experts_per_tok: config.num_experts_per_tok,
            norm_topk_prob: config.norm_topk_prob,
            router_aux_loss_coef: config.router_aux_loss_coef,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let (batch_size, seq_len, hidden_dim) = x.dims3()?;
        let num_tokens = batch_size * seq_len;
        let x_flat = x.reshape((num_tokens, hidden_dim))?;

        let logits = self.gate.forward(&x_flat)?;
        let probs = candle_nn::ops::softmax_last_dim(&logits)?;

        let probs_vec: Vec<f32> = probs.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1()?;
        let mut out_accum = Tensor::zeros((num_tokens, hidden_dim), x.dtype(), x.device())?;

        // For each token, select top-k experts
        for tok_idx in 0..num_tokens {
            let row_start = tok_idx * self.num_experts;
            let row_probs = &probs_vec[row_start..row_start + self.num_experts];

            let mut indexed_probs: Vec<(usize, f32)> = row_probs
                .iter()
                .copied()
                .enumerate()
                .collect();
            indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed_probs.truncate(self.num_experts_per_tok);

            let sum_prob: f32 = indexed_probs.iter().map(|p| p.1).sum();
            let tok_tensor = x_flat.narrow(0, tok_idx, 1)?;

            for (expert_idx, prob) in indexed_probs {
                let weight = if self.norm_topk_prob && sum_prob > 1e-20 {
                    prob / sum_prob
                } else {
                    prob
                };
                let expert_out = self.experts[expert_idx].forward(&tok_tensor)?;
                let weighted = (expert_out * (weight as f64))?;
                let prev_row = out_accum.narrow(0, tok_idx, 1)?;
                let new_row = (prev_row + weighted)?;

                let before = if tok_idx > 0 {
                    Some(out_accum.narrow(0, 0, tok_idx)?)
                } else {
                    None
                };
                let after = if tok_idx + 1 < num_tokens {
                    Some(out_accum.narrow(0, tok_idx + 1, num_tokens - tok_idx - 1)?)
                } else {
                    None
                };

                out_accum = match (before, after) {
                    (Some(b), Some(a)) => Tensor::cat(&[&b, &new_row, &a], 0)?,
                    (Some(b), None) => Tensor::cat(&[&b, &new_row], 0)?,
                    (None, Some(a)) => Tensor::cat(&[&new_row, &a], 0)?,
                    (None, None) => new_row,
                };
            }
        }

        let out = out_accum.reshape((batch_size, seq_len, hidden_dim))?;
        Ok((out, None))
    }
}

#[derive(Debug, Clone)]
pub enum Mlp {
    Dense(FeedForward),
    Moe(MoeFeedForward),
}

impl Mlp {
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        match self {
            Mlp::Dense(ffn) => Ok((ffn.forward(x)?, None)),
            Mlp::Moe(moe) => moe.forward(x),
        }
    }
}
