use crate::config::MiniMindConfig;
use crate::model::MiniMindForCausalLM;
use candle_core::{DType, Device, Result};
use candle_nn::VarBuilder;
use std::path::Path;

pub fn load_model_from_safetensors<P: AsRef<Path>>(
    weights_path: P,
    config: &MiniMindConfig,
    device: &Device,
    dtype: DType,
) -> Result<MiniMindForCausalLM> {
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path.as_ref()], dtype, device)?
    };
    MiniMindForCausalLM::load(config, vb)
}

pub fn create_empty_model(
    config: &MiniMindConfig,
    device: &Device,
    dtype: DType,
) -> Result<MiniMindForCausalLM> {
    let varmap = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, device);
    MiniMindForCausalLM::load(config, vb)
}
