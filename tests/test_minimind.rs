use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use minimind::{
    config::MiniMindConfig,
    generation::{GenerationConfig, Generator},
    lora::LoraLinear,
    rms_norm::RmsNorm,
    rope::RotaryEmbedding,
    MiniMindForCausalLM,
};

#[test]
fn test_config_defaults_and_serde() {
    let config = MiniMindConfig::default();
    assert_eq!(config.hidden_size, 768);
    assert_eq!(config.num_hidden_layers, 8);
    assert_eq!(config.num_attention_heads, 8);
    assert_eq!(config.num_key_value_heads, 4);
    assert_eq!(config.vocab_size, 6400);
    assert_eq!(config.head_dim(), 96);
    assert_eq!(config.num_rep(), 2);

    let json_str = serde_json::to_string(&config).expect("Serialize failed");
    let deserialized: MiniMindConfig = serde_json::from_str(&json_str).expect("Deserialize failed");
    assert_eq!(deserialized.hidden_size, config.hidden_size);
    assert_eq!(deserialized.num_hidden_layers, config.num_hidden_layers);
}

#[test]
fn test_rmsnorm() {
    let device = Device::Cpu;
    let dim = 32;
    let weight = Tensor::ones((dim,), DType::F32, &device).unwrap();
    let norm = RmsNorm::new(weight, 1e-6);

    let x = Tensor::randn(0.0f32, 1.0f32, (2, 4, dim), &device).unwrap();
    let y = norm.forward(&x).unwrap();

    assert_eq!(y.dims(), &[2, 4, dim]);

    let var = y.sqr().unwrap().mean_keepdim(candle_core::D::Minus1).unwrap();
    let var_val: f32 = var.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
    assert!((var_val - 1.0).abs() < 0.1);
}

#[test]
fn test_rope_embeddings() {
    let device = Device::Cpu;
    let dim = 64;
    let max_pos = 128;
    let rope = RotaryEmbedding::new(dim, max_pos, 1e6, false, &device).unwrap();

    assert_eq!(rope.freqs_cos.dims(), &[max_pos, dim]);
    assert_eq!(rope.freqs_sin.dims(), &[max_pos, dim]);

    let (cos, sin) = rope.get_embeddings(0, 16).unwrap();
    assert_eq!(cos.dims(), &[16, dim]);
    assert_eq!(sin.dims(), &[16, dim]);
}

#[test]
fn test_causal_lm_forward_and_loss() {
    let device = Device::Cpu;
    let mut config = MiniMindConfig::default();
    config.hidden_size = 64;
    config.num_hidden_layers = 2;
    config.num_attention_heads = 4;
    config.num_key_value_heads = 2;
    config.vocab_size = 128;
    config.max_position_embeddings = 64;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MiniMindForCausalLM::load(&config, vb).unwrap();

    let input_ids = Tensor::from_slice(&[1i64, 10, 20, 30, 40, 2], (1, 6), &device).unwrap();
    let labels = Tensor::from_slice(&[-100i64, 10, 20, 30, 40, 2], (1, 6), &device).unwrap();

    let out = model.forward(&input_ids, None, None, Some(&labels)).unwrap();

    assert_eq!(out.logits.dims(), &[1, 6, 128]);
    assert!(out.loss.is_some());

    let loss_val: f32 = out.loss.unwrap().to_scalar().unwrap();
    assert!(loss_val.is_finite());
    assert!(loss_val > 0.0);
}

#[test]
fn test_kv_cache_generation() {
    let device = Device::Cpu;
    let mut config = MiniMindConfig::default();
    config.hidden_size = 64;
    config.num_hidden_layers = 2;
    config.num_attention_heads = 4;
    config.num_key_value_heads = 2;
    config.vocab_size = 64;
    config.max_position_embeddings = 32;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MiniMindForCausalLM::load(&config, vb).unwrap();

    let gen_config = GenerationConfig {
        max_new_tokens: 5,
        temperature: 0.0, // greedy
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        eos_token_id: 2,
        do_sample: false,
    };
    let generator = Generator::new(&model, gen_config);

    let prompt = vec![1u32, 5, 12];
    let mut emitted = Vec::new();
    let generated = generator
        .generate(&prompt, |tok| {
            emitted.push(tok);
            true
        })
        .unwrap();

    assert_eq!(generated.len(), prompt.len() + emitted.len());
    assert!(!emitted.is_empty());
}

#[test]
fn test_lora_linear_forward_and_merge() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let base = candle_nn::linear_no_bias(32, 32, vb.pp("base")).unwrap();
    let lora = LoraLinear::new(base, 32, 32, 8, 16.0, vb.pp("lora")).unwrap();

    let x = Tensor::randn(0.0f32, 1.0f32, (1, 4, 32), &device).unwrap();
    let out = lora.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 4, 32]);

    let merged = lora.merge_weights().unwrap();
    assert_eq!(merged.dims(), &[32, 32]);
}
