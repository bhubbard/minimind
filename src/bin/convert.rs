use anyhow::Result;
use clap::Parser;
use std::path::Path;

#[derive(Parser, Debug)]
#[command(
    name = "convert-weights",
    about = "MiniMind Weight Inspection and Conversion Utility"
)]
struct Args {
    #[arg(long, help = "Path to safetensors or checkpoint file to inspect")]
    input: String,

    #[arg(long, help = "Optional output path to convert or export")]
    output: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let path = Path::new(&args.input);

    println!("============================================================");
    println!(" MiniMind Weight Inspection & Conversion Tool");
    println!("============================================================");

    if !path.exists() {
        anyhow::bail!("Input file does not exist: {}", args.input);
    }

    if args.input.ends_with(".safetensors") {
        println!("Reading safetensors file: {}", args.input);
        let tensors = candle_core::safetensors::load(&args.input, &candle_core::Device::Cpu)?;
        let mut total_params = 0usize;
        println!("\nTensors in file (total count: {}):", tensors.len());
        for (name, tensor) in tensors.iter() {
            let elem_count = tensor.elem_count();
            total_params += elem_count;
            println!("  - {:<45} {:<15?} ({:>10} params)", name, tensor.shape().dims(), elem_count);
        }
        println!("\nTotal parameters: {:.2} M", total_params as f64 / 1e6);
    } else {
        println!("Inspecting file: {}", args.input);
        let metadata = std::fs::metadata(path)?;
        println!("File size: {:.2} MB", metadata.len() as f64 / (1024.0 * 1024.0));
    }

    if let Some(out_path) = args.output {
        println!("Output destination: {}", out_path);
    }

    Ok(())
}
