//! MiniMind in Rust
//!
//! An educational, minimal, high-performance implementation of the MiniMind
//! Large Language Model architecture built on Hugging Face Candle.

pub mod attention;
pub mod block;
pub mod cache;
pub mod config;
pub mod dataset;
pub mod generation;
pub mod lora;
pub mod mlp;
pub mod model;
pub mod rms_norm;
pub mod rope;
pub mod trainer;
pub mod weights;

pub use config::MiniMindConfig;
pub use generation::{GenerationConfig, Generator};
pub use model::{CausalLmOutput, MiniMindForCausalLM, MiniMindModel};
