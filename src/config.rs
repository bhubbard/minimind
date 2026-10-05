use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiniMindConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default)]
    pub use_moe: bool,
    #[serde(default)]
    pub dropout: f64,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_bos_token_id")]
    pub bos_token_id: u32,
    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: u32,
    #[serde(default = "default_true")]
    pub flash_attn: bool,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub inference_rope_scaling: bool,

    // MoE specific configs
    #[serde(default = "default_num_experts")]
    pub num_experts: usize,
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    #[serde(default = "default_router_aux_loss_coef")]
    pub router_aux_loss_coef: f64,
}

fn default_hidden_size() -> usize { 768 }
fn default_num_hidden_layers() -> usize { 8 }
fn default_vocab_size() -> usize { 6400 }
fn default_bos_token_id() -> u32 { 1 }
fn default_eos_token_id() -> u32 { 2 }
fn default_true() -> bool { true }
fn default_num_attention_heads() -> usize { 8 }
fn default_num_key_value_heads() -> usize { 4 }
fn default_hidden_act() -> String { "silu".to_string() }
fn default_max_position_embeddings() -> usize { 32768 }
fn default_rms_norm_eps() -> f64 { 1e-6 }
fn default_rope_theta() -> f32 { 1e6 }
fn default_num_experts() -> usize { 4 }
fn default_num_experts_per_tok() -> usize { 1 }
fn default_router_aux_loss_coef() -> f64 { 5e-4 }

impl Default for MiniMindConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            num_hidden_layers: 8,
            use_moe: false,
            dropout: 0.0,
            vocab_size: 6400,
            bos_token_id: 1,
            eos_token_id: 2,
            flash_attn: true,
            num_attention_heads: 8,
            num_key_value_heads: 4,
            head_dim: None,
            hidden_act: "silu".to_string(),
            intermediate_size: None,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: 1e6,
            tie_word_embeddings: true,
            inference_rope_scaling: false,
            num_experts: 4,
            num_experts_per_tok: 1,
            moe_intermediate_size: None,
            norm_topk_prob: true,
            router_aux_loss_coef: 5e-4,
        }
    }
}

impl MiniMindConfig {
    pub fn head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn intermediate_size(&self) -> usize {
        if let Some(size) = self.intermediate_size {
            size
        } else {
            let pi = std::f64::consts::PI;
            ((self.hidden_size as f64 * pi / 64.0).ceil() as usize) * 64
        }
    }

    pub fn moe_intermediate_size(&self) -> usize {
        self.moe_intermediate_size.unwrap_or_else(|| self.intermediate_size())
    }

    pub fn num_rep(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}
