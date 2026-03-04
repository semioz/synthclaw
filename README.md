# synthclaw

Lightweight synthetic data generation in Rust. Generate and augment datasets using LLMs (OpenAI, Anthropic) with support for HuggingFace datasets.

Available as both a CLI tool and a Rust library.

## Installation

### CLI

```bash
cargo install --path .
```

### Library

Add to `Cargo.toml`:

```toml
[dependencies]
synth_claw = { git = "https://github.com/semioz/synthclaw" }
```

## CLI Usage

### Explore datasets

```bash
# Search HuggingFace
synthclaw datasets search "sentiment" --limit 10

# Get dataset info
synthclaw datasets info cornell-movie-review-data/rotten_tomatoes

# Preview rows
synthclaw datasets preview cornell-movie-review-data/rotten_tomatoes --rows 5 --split train
```

### Generate (dry-run only for now)

```bash
synthclaw generate --dry-run \
  --prompt "Generate a {category} product review" \
  --provider openai \
  --categories electronics,books,clothing \
  -n 100

# From config file
synthclaw generate --dry-run --config examples/configs/generate_reviews.yaml
```

## Library Usage

```rust
use synth_claw::{
    config::SynthConfig,
    datasets::{HuggingFaceSource, DataSource},
    providers::{create_provider, GenerationRequest},
};

// Load and query a HuggingFace dataset
let mut source = HuggingFaceSource::new(
    "cornell-movie-review-data/rotten_tomatoes".to_string(),
    None,
    "train".to_string(),
    None,
)?;
let records = source.load(Some(100))?;

// Create an LLM provider from config
let config = SynthConfig::from_file(&"config.yaml".into())?;
let provider = create_provider(&config.provider)?;

// Make a generation request
let response = provider.generate(GenerationRequest {
    prompt: "Generate a movie review".to_string(),
    system_prompt: None,
    temperature: Some(0.7),
    max_tokens: Some(500),
}).await?;

println!("{}", response.content);
```

## Configuration

Example config for generating from scratch:

```yaml
name: "product_reviews"

provider:
  type: openai
  model: "gpt-4o-mini"

generation:
  task: generate
  count: 500
  concurrency: 10
  categories:
    - electronics
    - books
    - clothing
  template: |
    Generate a realistic {category} product review.
    Output only the review text.

output:
  format: jsonl
  path: "./output/reviews.jsonl"
```

Example config for augmenting existing data:

```yaml
name: "sentiment_augmentation"

source:
  type: huggingface
  dataset: "cornell-movie-review-data/rotten_tomatoes"
  split: "train"
  sample: 1000

provider:
  type: anthropic
  model: "claude-3-5-sonnet-20241022"

generation:
  task: augment
  count_per_example: 3
  concurrency: 5
  strategy: paraphrase

output:
  format: parquet
  path: "./output/augmented.parquet"

checkpoint:
  enabled: true
```

## Environment variables

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```
