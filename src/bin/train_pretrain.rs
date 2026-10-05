use anyhow::Result;
use clap::Parser;
use minimind::{
    dataset::PretrainDataset,
    trainer::Trainer,
    MiniMindConfig,
};
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(
    name = "train-pretrain",
    about = "Pretrain MiniMind Language Model from scratch in Rust"
)]
struct Args {
    #[arg(long, default_value = "dataset/pretrain_t2t_mini.jsonl")]
    data_path: String,

    #[arg(long, default_value = "model/tokenizer.json")]
    tokenizer: String,

    #[arg(long, default_value = "out")]
    save_dir: String,

    #[arg(long, default_value_t = 2)]
    epochs: usize,

    #[arg(long, default_value_t = 4)]
    batch_size: usize,

    #[arg(long, default_value_t = 5e-4)]
    learning_rate: f64,

    #[arg(long, default_value_t = 768)]
    hidden_size: usize,

    #[arg(long, default_value_t = 8)]
    num_hidden_layers: usize,

    #[arg(long, default_value_t = 256)]
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
    println!(" MiniMind Rust Pretrainer");
    println!("============================================================");
    println!("Device: {:?}", device);

    let tokenizer = Tokenizer::from_file(Path::new(&args.tokenizer))
        .map_err(|e| anyhow::anyhow!("Tokenizer load failed: {}", e))?;

    let mut config = MiniMindConfig::default();
    config.hidden_size = args.hidden_size;
    config.num_hidden_layers = args.num_hidden_layers;
    config.max_position_embeddings = args.max_seq_len;

    println!(
        "Model: hidden_size={}, layers={}, vocab={}",
        config.hidden_size, config.num_hidden_layers, config.vocab_size
    );

    let varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let model = minimind::MiniMindForCausalLM::load(&config, vb)?;

    let dataset = if Path::new(&args.data_path).exists() {
        PretrainDataset::from_file(
            &args.data_path,
            tokenizer,
            args.max_seq_len,
            config.bos_token_id,
            config.eos_token_id,
            0,
        )?
    } else {
        println!("Dataset not found at '{}'. Creating dummy dataset for demonstration.", args.data_path);
        PretrainDataset {
            samples: vec![
                "Artificial intelligence and machine learning are revolutionizing software development.".to_string(),
                "The quick brown fox jumps over the lazy dog.".to_string(),
                "MiniMind is a minimal, educational language model built from scratch.".to_string(),
            ],
            tokenizer,
            max_length: args.max_seq_len,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
            pad_token_id: 0,
        }
    };

    let total_samples = dataset.len();
    let steps_per_epoch = (total_samples + args.batch_size - 1) / args.batch_size;
    let total_steps = steps_per_epoch * args.epochs;

    println!("Dataset samples: {}, Steps per epoch: {}, Total steps: {}", total_samples, steps_per_epoch, total_steps);

    let mut trainer = Trainer::new(varmap, args.learning_rate, total_steps, Some(1.0))?;

    std::fs::create_dir_all(&args.save_dir)?;

    let start_time = Instant::now();
    for epoch in 0..args.epochs {
        println!("\n--- Epoch [{}/{}] ---", epoch + 1, args.epochs);
        for step in 0..steps_per_epoch {
            let start_idx = step * args.batch_size;
            let end_idx = (start_idx + args.batch_size).min(total_samples);
            let indices: Vec<usize> = (start_idx..end_idx).collect();

            let (input_tensor, labels_tensor) = dataset.get_batch(&indices, &device)?;
            let out = model.forward(&input_tensor, None, None, Some(&labels_tensor))?;

            if let Some(loss) = out.loss {
                trainer.step_optimizer(&loss)?;

                if (step + 1) % args.log_interval == 0 || step + 1 == steps_per_epoch {
                    let loss_val: f32 = loss.to_scalar()?;
                    let current_lr = trainer.lr_scheduler.get_lr(trainer.current_step);
                    let elapsed = start_time.elapsed().as_secs_f64();
                    println!(
                        "Epoch [{}/{}], Step [{}/{}], Loss: {:.4}, LR: {:.8}, Time: {:.1}s",
                        epoch + 1, args.epochs, step + 1, steps_per_epoch, loss_val, current_lr, elapsed
                    );
                }
            }
        }

        let save_path = format!("{}/pretrain_epoch_{}.safetensors", args.save_dir, epoch + 1);
        trainer.save_checkpoint(&save_path)?;
        println!("Checkpoint saved: {}", save_path);
    }

    println!("\nPretraining completed in {:.2}s!", start_time.elapsed().as_secs_f64());
    Ok(())
}
