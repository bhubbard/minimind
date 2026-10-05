use anyhow::Result;
use clap::Parser;
use minimind::{
    weights::load_model_from_safetensors,
    GenerationConfig, Generator, MiniMindConfig, MiniMindForCausalLM,
};
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(
    name = "minimind-cli",
    about = "MiniMind LLM Inference, Chat, and Benchmarking CLI in Rust"
)]
struct Args {
    #[arg(long, default_value = "model/tokenizer.json")]
    tokenizer: String,

    #[arg(long, default_value = "model/config.json")]
    config: String,

    #[arg(long, default_value = "out/full_sft_768.safetensors")]
    weights: String,

    #[arg(long, default_value_t = 768)]
    hidden_size: usize,

    #[arg(long, default_value_t = 8)]
    num_hidden_layers: usize,

    #[arg(long, default_value_t = 0.85)]
    temperature: f64,

    #[arg(long, default_value_t = 0.95)]
    top_p: f64,

    #[arg(long, default_value_t = 50)]
    top_k: usize,

    #[arg(long, default_value_t = 512)]
    max_new_tokens: usize,

    #[arg(long, default_value_t = 1.0)]
    repetition_penalty: f64,

    #[arg(long, default_value = "cpu")]
    device: String,

    #[arg(long, default_value_t = 0)]
    mode: usize, // 0 = Automated Test, 1 = Interactive
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

    println!("============================================================");
    println!(" MiniMind Native Rust CLI & Inference Engine");
    println!("============================================================");

    let device = select_device(&args.device);
    println!("Running on device: {:?}", device);

    // Load Tokenizer
    let tokenizer_path = Path::new(&args.tokenizer);
    if !tokenizer_path.exists() {
        anyhow::bail!("Tokenizer file not found at: {}", args.tokenizer);
    }
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    println!("Tokenizer loaded successfully (vocab size: {})", tokenizer.get_vocab_size(true));

    // Load or create Config
    let config = if Path::new(&args.config).exists() {
        let content = std::fs::read_to_string(&args.config)?;
        serde_json::from_str::<MiniMindConfig>(&content).unwrap_or_else(|_| {
            let mut cfg = MiniMindConfig::default();
            cfg.hidden_size = args.hidden_size;
            cfg.num_hidden_layers = args.num_hidden_layers;
            cfg
        })
    } else {
        let mut cfg = MiniMindConfig::default();
        cfg.hidden_size = args.hidden_size;
        cfg.num_hidden_layers = args.num_hidden_layers;
        cfg
    };

    println!(
        "Model Config: hidden_size={}, layers={}, heads={}, intermediate={}",
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.intermediate_size()
    );

    // Load weights or initialize model
    let weights_path = Path::new(&args.weights);
    let model: MiniMindForCausalLM = if weights_path.exists() {
        println!("Loading weights from: {}", args.weights);
        load_model_from_safetensors(weights_path, &config, &device, candle_core::DType::F32)?
    } else {
        println!(
            "Weights '{}' not found. Initializing model with random weights for demo/testing.",
            args.weights
        );
        minimind::weights::create_empty_model(&config, &device, candle_core::DType::F32)?
    };

    let gen_config = GenerationConfig {
        max_new_tokens: args.max_new_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        repetition_penalty: args.repetition_penalty,
        eos_token_id: config.eos_token_id,
        do_sample: args.temperature > 0.0,
    };
    let generator = Generator::new(&model, gen_config);

    let prompts = vec![
        "What are your specialties?",
        "Why is the sky blue?",
        "Please write a Python function to calculate the Fibonacci sequence",
        "Explain the basic process of photosynthesis",
        "How should I prepare if it rains tomorrow?",
        "Compare the pros and cons of cats and dogs as pets",
        "Explain what machine learning is",
        "Recommend some delicious culinary specialties from around the world",
    ];

    let mode = if args.mode == 0 {
        print!("[0] Automated Test\n[1] Manual Interactive\nSelect mode (0/1): ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        input.trim().parse::<usize>().unwrap_or(0)
    } else {
        args.mode
    };

    if mode == 0 {
        println!("\nStarting Automated Benchmark Test...");
        for prompt in prompts {
            run_prompt(prompt, &tokenizer, &generator)?;
        }
    } else {
        println!("\nInteractive Chat Mode started (Ctrl+C to exit):");
        let stdin = io::stdin();
        let mut lines = stdin.lock().lines();
        loop {
            print!("\n💬: ");
            io::stdout().flush()?;
            if let Some(Ok(line)) = lines.next() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                run_prompt(trimmed, &tokenizer, &generator)?;
            } else {
                break;
            }
        }
    }

    Ok(())
}

fn run_prompt(
    prompt: &str,
    tokenizer: &Tokenizer,
    generator: &Generator,
) -> Result<()> {
    println!("\n💬: {}", prompt);
    let chat_formatted = format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", prompt);

    let encoding = tokenizer
        .encode(chat_formatted.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
    let prompt_tokens = encoding.get_ids();

    print!("🧠: ");
    io::stdout().flush()?;

    let start_time = Instant::now();
    let mut generated_count = 0usize;

    let mut token_buffer = Vec::new();

    let _output_ids = generator.generate(prompt_tokens, |token| {
        generated_count += 1;
        token_buffer.push(token);

        // Decode streaming token
        if let Ok(piece) = tokenizer.decode(&[token], false) {
            print!("{}", piece);
            let _ = io::stdout().flush();
        }
        true
    })?;

    let elapsed = start_time.elapsed().as_secs_f64();
    let speed = if elapsed > 0.0 {
        generated_count as f64 / elapsed
    } else {
        0.0
    };

    println!("\n[Speed]: {:.2} tokens/s ({} tokens in {:.2}s)\n", speed, generated_count, elapsed);

    Ok(())
}
