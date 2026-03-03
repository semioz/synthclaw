use clap::{Args, Subcommand};
use console::style;

#[derive(Subcommand)]
pub enum DatasetsCommand {
    Search(SearchArgs),
    Info(InfoArgs),
    Preview(PreviewArgs),
}

#[derive(Args)]
pub struct SearchArgs {
    pub query: String,
    #[arg(short, long, default_value = "10")]
    pub limit: usize,
}

#[derive(Args)]
pub struct InfoArgs {
    /// id like "cornell-movie-review-data/rotten_tomatoes"
    pub dataset: String,
}

#[derive(Args)]
pub struct PreviewArgs {
    pub dataset: String,
    /// dataset subset/config (if any)
    #[arg(short, long)]
    pub subset: Option<String>,
    #[arg(short = 'S', long, default_value = "train")]
    pub split: String,
    #[arg(short, long, default_value = "5")]
    pub rows: usize,
}

pub async fn run(cmd: DatasetsCommand) -> anyhow::Result<()> {
    match cmd {
        DatasetsCommand::Search(args) => search(args).await,
        DatasetsCommand::Info(args) => info(args).await,
        DatasetsCommand::Preview(args) => preview(args).await,
    }
}

async fn search(args: SearchArgs) -> anyhow::Result<()> {
    println!(
        "{} Searching for datasets matching: {}",
        style("→").cyan().bold(),
        style(&args.query).green()
    );

    // TODO: HF dataset search
    println!(
        "{}",
        style("Dataset search not yet implemented (Phase 3)").yellow()
    );

    Ok(())
}

async fn info(args: InfoArgs) -> anyhow::Result<()> {
    println!(
        "{} Getting info for: {}",
        style("→").cyan().bold(),
        style(&args.dataset).green()
    );

    // TODO: HF dataset info
    println!(
        "{}",
        style("Dataset info not yet implemented (Phase 3)").yellow()
    );

    Ok(())
}

async fn preview(args: PreviewArgs) -> anyhow::Result<()> {
    println!(
        "{} Previewing: {} (split: {}, rows: {})",
        style("→").cyan().bold(),
        style(&args.dataset).green(),
        args.split,
        args.rows
    );

    println!(
        "{}",
        style("Dataset preview not yet implemented (Phase 3)").yellow()
    );

    Ok(())
}
