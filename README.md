# synthclaw

Lightweight synthetic data generation in Rust. Generate and augment datasets using LLMs (OpenAI, Anthropic) with support for HuggingFace datasets.

Available as both a CLI tool and a Rust library.

## Installation

### CLI

```bash
cargo install --path .
```

### Library

```toml
[dependencies]
synth_claw = { git = "https://github.com/semioz/synthclaw" }
```

## Quick Start

```bash
export OPENAI_API_KEY=sk-...

# Generate 50 product reviews across categories
synthclaw generate \
  --prompt "Generate a realistic {category} product review, 2-3 sentences" \
  --provider openai \
  --categories electronics,books,clothing \
  -n 50 \
  -o reviews.jsonl

# Or use a config file
synthclaw generate --config examples/configs/generate_reviews.yaml
```

## CLI Usage

### Explore HuggingFace Datasets

```bash
# Search
synthclaw datasets search "sentiment" --limit 10

# Get info
synthclaw datasets info cornell-movie-review-data/rotten_tomatoes

# Preview rows
synthclaw datasets preview cornell-movie-review-data/rotten_tomatoes --rows 5
```

### Generate Data

```bash
# From scratch with categories
synthclaw generate \
  --prompt "Generate a {category} example" \
  --provider openai \
  --categories positive,negative \
  -n 100

# Dry run (no API calls)
synthclaw generate --dry-run --config config.yaml
```

## Writing Good Prompts

The tool uses **system prompts** by default to ensure clean outputs. You provide the **user prompt template**.

### Template Variables

For **generate** mode:
- `{category}` - current category being generated
- `{index}` - item number (0, 1, 2...)

For **augment** mode:
- Any column from source data: `{text}`, `{label}`, etc.

### Good Prompt Examples

**Product Reviews:**
```yaml
template: |
  Generate a realistic product review for: {category}
  
  Requirements:
  - Customer perspective, 2-4 sentences
  - Include specific details (brand, features, price)
  - Natural tone - can be positive, negative, or mixed
```

**Sentiment Data:**
```yaml
template: |
  Generate a {category} movie review.
  
  Requirements:
  - The sentiment must clearly be {category}
  - 1-3 sentences
  - Mention specific aspects (acting, plot, visuals)
```

**Data Augmentation (paraphrase):**
```yaml
template: |
  Paraphrase this text while preserving meaning and sentiment:
  
  Original: {text}
  
  Paraphrase:
```

**Question-Answer Generation:**
```yaml
template: |
  Based on this document, generate a Q&A pair:
  
  Document: {text}
  
  Output JSON: {"question": "...", "answer": "..."}
system_prompt: |
  Generate educational Q&A pairs. Output ONLY valid JSON.
```

### Bad Prompts (avoid these)

```yaml
# Too vague - LLM will add explanations
template: "Generate something for {category}"

# Missing variable
template: "Generate a review"  # error: no {category} but categories specified

# Wrong variable for mode
template: "Paraphrase {text}"  # error in generate mode: {text} not available
```

## Configuration

### Generate from Scratch

```yaml
name: "product_reviews"

provider:
  type: openai
  model: "gpt-4o-mini"
  temperature: 0.8

generation:
  task: generate
  count: 100
  concurrency: 10
  categories:
    - electronics
    - books
    - clothing
  template: |
    Generate a realistic {category} product review.
    2-3 sentences, customer perspective, specific details.

output:
  format: jsonl
  path: "./output/reviews.jsonl"
```

### Augment Existing Data

```yaml
name: "sentiment_augmentation"

source:
  type: huggingface
  dataset: "cornell-movie-review-data/rotten_tomatoes"
  split: "train"
  sample: 500

provider:
  type: openai
  model: "gpt-4o-mini"

generation:
  task: augment
  count_per_example: 2
  concurrency: 10
  strategy: paraphrase

output:
  format: jsonl
  path: "./output/augmented.jsonl"
```

### Custom System Prompt

Override the default system prompt when you need specific behavior:

```yaml
generation:
  template: |
    Generate a {category} example in JSON format.
  system_prompt: |
    You are a data generation assistant.
    Output ONLY valid JSON, no markdown, no explanations.
    Schema: {"text": "...", "label": "..."}
```

## Library Usage

```rust
use synth_claw::{
    config::SynthConfig,
    datasets::{HuggingFaceSource, DataSource},
    providers::{create_provider, GenerationRequest},
};

// Load HuggingFace dataset
let mut source = HuggingFaceSource::new(
    "cornell-movie-review-data/rotten_tomatoes".to_string(),
    None,
    "train".to_string(),
    None,
)?;
let records = source.load(Some(100))?;

// Create provider and generate
let config = SynthConfig::from_file(&"config.yaml".into())?;
let provider = create_provider(&config.provider)?;

let response = provider.generate(GenerationRequest {
    prompt: "Generate a movie review".to_string(),
    system_prompt: Some("Output only the review text.".to_string()),
    temperature: Some(0.7),
    max_tokens: Some(500),
}).await?;
```

## Output Formats

- `jsonl` - Line-delimited JSON (recommended for large datasets)
- `csv` - Comma-separated values
- `parquet` - Apache Parquet (efficient for analytics)

## Environment Variables

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## Project Structure

```
src/
  cli/           CLI commands
  config/        YAML config schema
  datasets/      Data sources (HuggingFace, local files)
  providers/     LLM providers (OpenAI, Anthropic)
  generation/    Generation engine
  output/        Output writers (JSONL, CSV, Parquet)
```
