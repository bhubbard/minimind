use candle_core::{Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

#[derive(Debug, Clone)]
pub struct LoraLinear {
    pub base: Linear,
    pub a: Linear,
    pub b: Linear,
    pub scale: f64,
    pub rank: usize,
}

impl LoraLinear {
    pub fn new(base: Linear, in_features: usize, out_features: usize, rank: usize, alpha: f64, vb: VarBuilder) -> Result<Self> {
        let a = linear_no_bias(in_features, rank, vb.pp("a"))?;
        let b = linear_no_bias(rank, out_features, vb.pp("b"))?;
        let scale = alpha / rank as f64;
        Ok(Self {
            base,
            a,
            b,
            scale,
            rank,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let base_out = self.base.forward(x)?;
        let a_out = self.a.forward(x)?;
        let b_out = self.b.forward(&a_out)?;
        let lora_out = (b_out * self.scale)?;
        base_out + lora_out
    }

    pub fn merge_weights(&self) -> Result<Tensor> {
        let base_weight = self.base.weight();
        let b_weight = self.b.weight();
        let a_weight = self.a.weight();
        let delta = b_weight.matmul(a_weight)?;
        let scaled_delta = (delta * self.scale)?;
        base_weight + scaled_delta
    }
}
