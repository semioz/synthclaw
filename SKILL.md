# Skill: synthclaw - Synthetic Data Generation

## Description
Generate synthetic data or augment existing datasets using LLMs (OpenAI, Anthropic). Supports HuggingFace datasets, multiple output formats (JSONL, CSV, Parquet).

## When to Use
- User needs synthetic training data for ML models
- User wants to augment/expand an existing dataset
- User needs test data with specific patterns or categories
- User wants to paraphrase or transform text data
- User needs to explore HuggingFace datasets

## Prerequisites
- `cargo install synthclaw` (or available in PATH)
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` environment variable set

## CLI Commands

### Generate Data from Scratch
```bash
synthclaw generate \
  --provider openai \
  --model gpt-4o-mini \
  --prompt "Generate a realistic {category} product review, 2-3 sentences" \
  --categories "positive,negative,neutral" \
  -n 100 \
  --concurrency 10 \
  -o output.jsonl
```

### Generate with Config File
```bash
synthclaw generate --config config.yaml
```

### Dry Run (Preview without API calls)
```bash
synthclaw generate --config config.yaml --dry-run
```

### Search HuggingFace Datasets
```bash
synthclaw datasets search "sentiment" --limit 10
```

### Get Dataset Info
```bash
synthclaw datasets info cornell-movie-review-data/rotten_tomatoes
```

### Preview Dataset Rows
```bash
synthclaw datasets preview cornell-movie-review-data/rotten_tomatoes --rows 5
```

## Config File Examples

### Generate from Scratch
```yaml
name: "product_reviews"

provider:
  type: openai
  model: "gpt-4o-mini"

generation:
  task: generate
  count: 100
  concurrency: 10
  categories:
    - positive
    - negative
    - neutral
  template: |
    Generate a realistic {category} product review for an electronic device.
    Keep it to 2-3 sentences. Output only the review text.

output:
  format: jsonl
  path: "./output/reviews.jsonl"
```

### Augment HuggingFace Dataset
```yaml
name: "sentiment_augmentation"

source:
  type: huggingface
  dataset: "cornell-movie-review-data/rotten_tomatoes"
  split: "train"
  sample: 100

provider:
  type: openai
  model: "gpt-4o-mini"

generation:
  task: augment
  count_per_example: 2
  concurrency: 10
  strategy: paraphrase
  template: |
    Paraphrase this movie review while preserving its sentiment:
    
    Original: {text}
    
    Paraphrased:

output:
  format: jsonl
  path: "./output/augmented.jsonl"
```

### Augment Local File
```yaml
name: "local_augmentation"

source:
  type: local
  path: "./data/input.jsonl"
  format: jsonl
  sample: 50

provider:
  type: anthropic
  model: "claude-haiku-4-5-20251001"

generation:
  task: augment
  count_per_example: 3
  concurrency: 5
  template: |
    Rewrite this text in a different style:
    
    Original: {content}
    
    Rewritten:

output:
  format: jsonl
  path: "./output/augmented.jsonl"
```

## Template Variables

### Generate Mode
- `{category}` - Current category being generated
- `{index}` - Item number (0, 1, 2...)

### Augment Mode
- Any column from source data: `{text}`, `{label}`, `{content}`, etc.

## Output Formats
- `jsonl` - Line-delimited JSON (recommended)
- `csv` - Comma-separated values
- `parquet` - Apache Parquet (efficient for large datasets)

## Output Schema
Each output record contains:
```json
{
  "content": "Generated or augmented text",
  "category": "positive",
  "source_index": 42,
  "input_tokens": 150,
  "output_tokens": 45
}
```

## Providers
- `openai` - Models: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo, o1, o1-mini, o3-mini
- `anthropic` - Models: claude-sonnet-4-20250514, claude-haiku-4-5-20251001, claude-opus-4-5-20251101

## Common Workflows

### 1. Generate Classification Training Data
```bash
synthclaw generate \
  --provider openai \
  --prompt "Generate a {category} customer support ticket. Include the issue description and customer tone." \
  --categories "billing,technical,shipping,returns" \
  -n 200 \
  -o support_tickets.jsonl
```

### 2. Expand Small Dataset
```yaml
# First, check what columns exist
synthclaw datasets preview my-dataset --rows 1

# Then augment
name: "expand_dataset"
source:
  type: huggingface
  dataset: "my-small-dataset"
  split: "train"
generation:
  task: augment
  count_per_example: 5
  strategy: paraphrase
output:
  format: parquet
  path: "./expanded.parquet"
```

### 3. Generate Test Fixtures
```bash
synthclaw generate \
  --provider openai \
  --prompt "Generate a realistic {category} JSON object for an e-commerce API test" \
  --categories "product,order,user,review" \
  -n 20 \
  -o test_fixtures.jsonl
```

## Tips
- Use `--dry-run` to verify config before making API calls
- Higher `concurrency` = faster but more API rate limit risk (10-20 is good)
- Use `sample` in source config to test with subset first
- Check dataset columns with `datasets preview` before writing augment templates
