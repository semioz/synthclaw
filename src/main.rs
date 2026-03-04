use clap::{Parser, Subcommand};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

mod cli;

use cli::{datasets, estimate, generate, resume};

#[derive(Parser)]
#[command(name = "synthclaw")]
#[command(author, version, about = "Lightweight synthetic data generation CLI")]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate synthetic data from scratch or augment existing datasets
    Generate(generate::GenerateArgs),
    /// Search, explore and preview HuggingFace datasets
    #[command(subcommand)]
    Datasets(datasets::DatasetsCommand),
    /// Resume an interrupted generation from checkpoint
    Resume(resume::ResumeArgs),
    /// Estimate cost for a generation job without running it
    Estimate(estimate::EstimateArgs),
}

fn setup_logging(verbose: bool) {
    let filter = if verbose {
        EnvFilter::new("synth_claw=debug,info")
    } else {
        EnvFilter::new("synth_claw=info,warn")
    };

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(false))
        .with(filter)
        .init();
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    setup_logging(cli.verbose);

    match cli.command {
        Commands::Generate(args) => generate::run(args).await?,
        Commands::Datasets(cmd) => datasets::run(cmd).await?,
        Commands::Resume(args) => resume::run(args).await?,
        Commands::Estimate(args) => estimate::run(args).await?,
    }

    Ok(())
}
