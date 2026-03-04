use clap::{Args, Subcommand};
use console::style;
use synth_claw::datasets::hf::{get_dataset_info, preview_dataset, search_datasets};

#[derive(Subcommand)]
pub enum DatasetsCommand {
    /// Search for datasets on Hugging Face
    Search(SearchArgs),
    /// Get information about a specific dataset
    Info(InfoArgs),
    /// Preview first few rows of a dataset
    Preview(PreviewArgs),
}

#[derive(Args)]
pub struct SearchArgs {
    /// Search query
    pub query: String,
    /// Maximum number of results
    #[arg(short, long, default_value = "10")]
    pub limit: usize,
}

#[derive(Args)]
pub struct InfoArgs {
    /// Dataset identifier (e.g., "cornell-movie-review-data/rotten_tomatoes")
    pub dataset: String,
}

#[derive(Args)]
pub struct PreviewArgs {
    /// Dataset identifier
    pub dataset: String,
    /// Dataset subset/config (if any)
    #[arg(short, long)]
    pub subset: Option<String>,
    /// Split to preview
    #[arg(short = 'S', long, default_value = "train")]
    pub split: String,
    /// Number of rows to show
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

    let results = search_datasets(&args.query, args.limit).await?;

    if results.is_empty() {
        println!("{}", style("No datasets found").yellow());
        return Ok(());
    }

    println!("\n{}", style("Results:").bold());
    for result in results {
        println!(
            "  {} {} (↓{} ♥{})",
            style("•").cyan(),
            style(&result.id).green(),
            result.downloads,
            result.likes
        );
    }

    Ok(())
}

async fn info(args: InfoArgs) -> anyhow::Result<()> {
    println!(
        "{} Getting info for: {}",
        style("→").cyan().bold(),
        style(&args.dataset).green()
    );

    let info = get_dataset_info(&args.dataset).await?;

    println!("\n{}", style("Dataset Info:").bold());
    println!("  Name: {}", style(&info.name).green());
    if info.description.as_ref().is_some_and(|d| !d.is_empty()) {
        println!("  Description: {}", info.description.as_ref().unwrap());
    }
    println!("  Total rows: {}", info.num_rows);
    println!("  Columns: {}", info.columns.join(", "));

    println!("\n{}", style("Splits:").bold());
    for split in &info.splits {
        println!("  {} ({} rows)", style(&split.name).cyan(), split.num_rows);
    }

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

    let rows = preview_dataset(&args.dataset, args.subset.as_deref(), &args.split, args.rows).await?;

    println!("\n{}", style("Preview:").bold());
    for (i, row) in rows.iter().enumerate() {
        println!("\n{}:", style(format!("Row {}", i + 1)).cyan().bold());
        if let Some(obj) = row.as_object() {
            for (key, value) in obj {
                let value_str = match value {
                    serde_json::Value::String(s) => {
                        if s.len() > 100 {
                            format!("{}...", &s[..100])
                        } else {
                            s.clone()
                        }
                    }
                    other => other.to_string(),
                };
                println!("  {}: {}", style(key).yellow(), value_str);
            }
        }
    }

    Ok(())
}
