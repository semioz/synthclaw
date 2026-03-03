use clap::Args;
use console::style;
use std::path::PathBuf;
use synth_claw::config::SynthConfig;

#[derive(Args)]
pub struct GenerateArgs {
    #[arg(short, long, conflicts_with_all = ["prompt", "categories"])]
    pub config: Option<PathBuf>,
    #[arg(short, long, requires = "provider")]
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
    } else if let Some(prompt) = &args.prompt {
        build_config_from_args(&args, prompt)?
    } else {
        anyhow::bail!("Either --config or --prompt must be provided");
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

    // TODO
    println!(
        "{}",
        style("Generation engine not yet implemented (Phase 4)").yellow()
    );

    Ok(())
}

fn build_config_from_args(args: &GenerateArgs, prompt: &str) -> anyhow::Result<SynthConfig> {
    use synth_claw::config::*;

    let provider_str = args
        .provider
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--provider is required when using --prompt"))?;

    let model = args
        .model
        .clone()
        .unwrap_or_else(|| match provider_str.as_str() {
            "openai" => "gpt-4o-mini".to_string(),
            "anthropic" => "claude-4-6-sonnet".to_string(),
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
            template: Some(prompt.to_string()),
            system_prompt: None,
            categories: args.categories.clone(),
        },
        output: OutputConfig {
            format: OutputFormat::Jsonl,
            path: output_path,
            batch_size: 100,
        },
        checkpoint: CheckpointConfig::default(),
    })
}
