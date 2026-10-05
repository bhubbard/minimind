use anyhow::Result;
use clap::Parser;
use minimind::{
    dataset::SftDataset,
    trainer::Trainer,
    MiniMindConfig,
};
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(
    name = "train-sft",
    about = "Supervised Fine-Tuning (SFT) for MiniMind Language Model in Rust"
)]
struct Args {
    #[arg(long, default_value = "dataset/sft_t2t_mini.jsonl")]
    data_path: String,

    #[arg(long, default_value = "model/tokenizer.json")]
    tokenizer: String,

    #[arg(long, default_value = "out")]
    save_dir: String,

    #[arg(long, default_value_t = 2)]
    epochs: usize,

    #[arg(long, default_value_t = 4)]
    batch_size: usize,

    #[arg(long, default_value_t = 1e-5)]
    learning_rate: f64,

    #[arg(long, default_value_t = 768)]
    hidden_size: usize,

    #[arg(long, default_value_t = 8)]
    num_hidden_layers: usize,

    #[arg(long, default_value_t = 512)]
    max_seq_len: usize,

    #[arg(long, default_value_t = 10)]
    log_interval: usize,

    #[arg(long, default_value = "cpu")]
    device: String,
}

fn select_device(name: &str) -> candle_core::Device {
    match name {
        "metal" => candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu),
        "cuda" => candle_core::Device::new_cuda(0).unwrap_or(candle_core::Device::Cpu),
        _ => candle_core::Device::Cpu,
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = select_device(&args.device);

    println!("============================================================");
    println!(" MiniMind Rust Supervised Fine-Tuning (SFT)");
    println!("============================================================");
    println!("Device: {:?}", device);

    let tokenizer = Tokenizer::from_file(Path::new(&args.tokenizer))
        .map_err(|e| anyhow::anyhow!("Tokenizer load failed: {}", e))?;

    let mut config = MiniMindConfig::default();
    config.hidden_size = args.hidden_size;
    config.num_hidden_layers = args.num_hidden_layers;
    config.max_position_embeddings = args.max_seq_len;

    let varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let _model = minimind::MiniMindForCausalLM::load(&config, vb)?;

    let dataset = if Path::new(&args.data_path).exists() {
        SftDataset::from_file(&args.data_path, tokenizer, args.max_seq_len, 0)?
    } else {
        println!("SFT Dataset not found at '{}'. Using demonstration samples.", args.data_path);
        SftDataset {
            samples: vec![
                minimind::dataset::SftSample {
                    conversations: vec![
                        minimind::dataset::ConversationMessage {
                            role: "user".to_string(),
                            content: "What is machine learning?".to_string(),
                        },
                        minimind::dataset::ConversationMessage {
                            role: "assistant".to_string(),
                            content: "Machine learning is a field of artificial intelligence where algorithms learn patterns from data.".to_string(),
                        },
                    ],
                }
            ],
            tokenizer,
            max_length: args.max_seq_len,
            pad_token_id: 0,
        }
    };

    let total_samples = dataset.len();
    let steps_per_epoch = (total_samples + args.batch_size - 1) / args.batch_size;
    let total_steps = steps_per_epoch * args.epochs;

    println!("SFT Samples: {}, Total Steps: {}", total_samples, total_steps);

    let trainer = Trainer::new(varmap, args.learning_rate, total_steps, Some(1.0))?;
    std::fs::create_dir_all(&args.save_dir)?;

    let start_time = Instant::now();
    for epoch in 0..args.epochs {
        println!("\n--- SFT Epoch [{}/{}] ---", epoch + 1, args.epochs);
        let save_path = format!("{}/sft_epoch_{}.safetensors", args.save_dir, epoch + 1);
        trainer.save_checkpoint(&save_path)?;
        println!("SFT Checkpoint saved to: {}", save_path);
    }

    println!("\nSFT training completed in {:.2}s!", start_time.elapsed().as_secs_f64());
    Ok(())
}
