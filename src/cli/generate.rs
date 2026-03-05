use clap::Args;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use synth_claw::config::SynthConfig;
use synth_claw::generation::GenerationEngine;
use synth_claw::output::create_writer;

#[derive(Args)]
pub struct GenerateArgs {
    #[arg(short, long, conflicts_with_all = ["prompt", "categories"])]
    pub config: Option<PathBuf>,
    #[arg(short, long)]
    pub prompt: Option<String>,
    #[arg(long, value_delimiter = ',')]
    pub categories: Option<Vec<String>>,
    #[arg(short = 'n', long, default_value = "100")]
    pub count: usize,
    #[arg(long)]
    pub provider: Option<String>,
    #[arg(short, long)]
    pub model: Option<String>,
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    #[arg(long, default_value = "5")]
    pub concurrency: usize,

    /// show what would be generated without calling LLM
    #[arg(long)]
    pub dry_run: bool,
}

pub async fn run(args: GenerateArgs) -> anyhow::Result<()> {
    let config = if let Some(config_path) = &args.config {
        tracing::info!("Loading config from {:?}", config_path);
        SynthConfig::from_file(config_path)?
    } else if args.provider.is_some() {
        build_config_from_args(&args)?
    } else {
        anyhow::bail!("Either --config or --provider must be provided");
    };

    if args.dry_run {
        println!("{}", style("Dry run mode - no API calls will be made").yellow());
        println!("\n{}", style("Configuration:").bold());
        println!("  Name: {}", config.name);
        println!("  Task: {:?}", config.generation.task);
        println!("  Count: {}", config.generation.count);
        println!("  Concurrency: {}", config.generation.concurrency);
        println!("  Output: {:?}", config.output.path);
        if let Some(categories) = &config.generation.categories {
            println!("  Categories: {:?}", categories);
        }
        return Ok(());
    }

    println!(
        "{} Starting generation: {}",
        style("→").cyan().bold(),
        config.name
    );

    run_generation(&config).await?;

    Ok(())
}

fn build_config_from_args(args: &GenerateArgs) -> anyhow::Result<SynthConfig> {
    use synth_claw::config::*;

    let provider_str = args
        .provider
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--provider is required"))?;

    let model = args
        .model
        .clone()
        .unwrap_or_else(|| match provider_str.as_str() {
            "openai" => "gpt-4o-mini".to_string(),
            "anthropic" => "claude-haiku-4-5-20251001".to_string(),
            _ => "gpt-4o-mini".to_string(),
        });

    let provider = match provider_str.as_str() {
        "openai" => ProviderConfig::OpenAI {
            model,
            api_key: None,
            base_url: None,
            temperature: None,
            max_tokens: None,
        },
        "anthropic" => ProviderConfig::Anthropic {
            model,
            api_key: None,
            temperature: None,
            max_tokens: None,
        },
        other => anyhow::bail!("Unknown provider: {}", other),
    };

    let output_path = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from("./output/generated.jsonl"));

    Ok(SynthConfig {
        name: "cli-generation".to_string(),
        source: None,
        provider,
        generation: GenerationConfig {
            task: GenerationTask::Generate,
            count: args.count,
            count_per_example: None,
            concurrency: args.concurrency,
            strategy: None,
            strategy_config: Default::default(),
            template: args.prompt.clone(),
            system_prompt: None,
            categories: args.categories.clone(),
        },
        output: OutputConfig {
            format: OutputFormat::Jsonl,
            path: output_path,
            batch_size: 100,
        },
    })
}

async fn run_generation(config: &SynthConfig) -> anyhow::Result<()> {
    let engine = GenerationEngine::new(&config.provider, config.generation.clone())?;
    let stats = engine.stats();
    
    let total = match &config.generation.task {
        synth_claw::config::GenerationTask::Generate => config.generation.count,
        synth_claw::config::GenerationTask::Augment => {
            config.generation.count_per_example.unwrap_or(1) * config.generation.count
        }
    };

    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
            .progress_chars("#>-"),
    );

    println!("  Provider: {}", engine.provider().name());
    println!("  Model: {}", engine.provider().model());
    println!("  Concurrency: {}", config.generation.concurrency);
    println!();

    let results = engine.run(config).await?;
    
    pb.finish_and_clear();

    let mut writer = create_writer(&config.output.format, config.output.path.clone())?;
    for result in &results {
        writer.write(result)?;
    }
    writer.flush()?;

    let snapshot = stats.snapshot();
    println!("{}", style("Summary:").bold());
    println!("  Completed: {}", style(snapshot.completed).green());
    println!("  Failed: {}", style(snapshot.failed).red());
    println!("  Input tokens: {}", snapshot.total_input_tokens);
    println!("  Output tokens: {}", snapshot.total_output_tokens);
    
    let cost = engine.provider().estimate_cost(
        snapshot.total_input_tokens as u32,
        snapshot.total_output_tokens as u32,
    );
    println!("  Estimated cost: ${:.4}", cost);
    println!("\n  Output: {}", style(config.output.path.display()).cyan());

    Ok(())
}
