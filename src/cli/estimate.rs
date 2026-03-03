use clap::Args;
use console::style;
use std::path::PathBuf;
use synth_claw::config::SynthConfig;

#[derive(Args)]
pub struct EstimateArgs {
    #[arg(short, long)]
    pub config: PathBuf,
    #[arg(long)]
    pub detailed: bool,
}

pub async fn run(args: EstimateArgs) -> anyhow::Result<()> {
    let config = SynthConfig::from_file(&args.config)?;

    println!(
        "{} Cost estimate for: {}",
        style("→").cyan().bold(),
        style(&config.name).green()
    );

    println!("\n{}", style("Configuration Summary:").bold());
    println!("  Task: {:?}", config.generation.task);
    println!("  Count: {}", config.generation.count);

    if let Some(count_per) = config.generation.count_per_example {
        println!("  Per-example count: {}", count_per);
    }

    // TODO: actual cost estimation
    println!(
        "\n{}",
        style("Cost estimation not yet implemented)").yellow()
    );
    println!("Will calculate:");
    println!("  - Estimated input tokens");
    println!("  - Estimated output tokens");
    println!("  - Total API calls");
    println!("  - Estimated cost (USD)");

    Ok(())
}
