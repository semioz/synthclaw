use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthConfig {
    pub name: String,
    #[serde(default)]
    pub source: Option<SourceConfig>,
    pub provider: ProviderConfig,
    pub generation: GenerationConfig,
    pub output: OutputConfig,
    #[serde(default)]
    pub validation: Option<ValidationConfig>,
    #[serde(default)]
    pub hub: Option<HubConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SourceConfig {
    HuggingFace {
        dataset: String,
        #[serde(default)]
        subset: Option<String>,
        #[serde(default = "default_split")]
        split: String,
        #[serde(default)]
        sample: Option<usize>,
        #[serde(default)]
        columns: Option<Vec<String>>,
    },
    Local {
        path: PathBuf,
        format: FileFormat,
        #[serde(default)]
        sample: Option<usize>,
    },
}

fn default_split() -> String {
    "train".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FileFormat {
    Json,
    Jsonl,
    Csv,
    Parquet,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ProviderConfig {
    OpenAI {
        model: String,
        #[serde(default)]
        api_key: Option<String>,
        #[serde(default)]
        base_url: Option<String>,
        #[serde(default)]
        temperature: Option<f32>,
        #[serde(default)]
        max_tokens: Option<u32>,
    },
    Anthropic {
        model: String,
        #[serde(default)]
        api_key: Option<String>,
        #[serde(default)]
        temperature: Option<f32>,
        #[serde(default)]
        max_tokens: Option<u32>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub task: GenerationTask,
    #[serde(default = "default_count")]
    pub count: usize,
    #[serde(default)]
    pub count_per_example: Option<usize>,
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    #[serde(default)]
    pub strategy: Option<GenerationStrategy>,
    #[serde(default)]
    pub strategy_config: HashMap<String, serde_yaml::Value>,
    #[serde(default)]
    pub template: Option<String>,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub categories: Option<Vec<String>>,
}

fn default_count() -> usize {
    100
}

fn default_concurrency() -> usize {
    5
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationTask {
    Generate,
    Augment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationStrategy {
    Paraphrase,
    StyleTransfer,
    BackTranslation,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub format: OutputFormat,
    pub path: PathBuf,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
}

fn default_batch_size() -> usize {
    100
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    Json,
    Jsonl,
    Csv,
    Parquet,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ValidationConfig {
    #[serde(default)]
    pub min_length: Option<usize>,
    #[serde(default)]
    pub max_length: Option<usize>,
    #[serde(default)]
    pub json: bool,
    #[serde(default)]
    pub json_schema: Option<Vec<String>>,
    #[serde(default)]
    pub blocklist: bool,
    #[serde(default)]
    pub repetition: bool,
    #[serde(default)]
    pub dedupe: Option<DedupeStrategy>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DedupeStrategy {
    Exact,
    Normalized,
    Jaccard,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HubConfig {
    #[serde(default)]
    pub token: Option<String>,
    #[serde(default)]
    pub repo: Option<String>,
    #[serde(default)]
    pub private: bool,
}

impl SynthConfig {
    pub fn from_yaml(content: &str) -> crate::Result<Self> {
        serde_yaml::from_str(content).map_err(Into::into)
    }

    pub fn from_file(path: &PathBuf) -> crate::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_yaml(&content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_augment_config() {
        let yaml = r#"
name: "sentiment_augmentation"

source:
  type: huggingface
  dataset: "cornell-movie-review-data/rotten_tomatoes"
  split: "train"
  sample: 1000

provider:
  type: openai
  model: "gpt-4o-mini"

generation:
  task: augment
  count_per_example: 3
  concurrency: 10
  strategy: paraphrase

output:
  format: jsonl
  path: "./output/augmented.jsonl"
"#;

        let config = SynthConfig::from_yaml(yaml).unwrap();
        assert_eq!(config.name, "sentiment_augmentation");
        assert!(matches!(
            config.source,
            Some(SourceConfig::HuggingFace { .. })
        ));
        assert!(matches!(config.provider, ProviderConfig::OpenAI { .. }));
        assert!(matches!(config.generation.task, GenerationTask::Augment));
    }

    #[test]
    fn test_parse_generate_config() {
        let yaml = r#"
name: "product_reviews"

provider:
  type: anthropic
  model: "claude-haiku-4-5-20251001"

generation:
  task: generate
  count: 500
  concurrency: 5
  categories:
    - electronics
    - books
    - clothing
  template: |
    Generate a realistic {category} product review.
    Output only the review text.

output:
  format: parquet
  path: "./output/reviews.parquet"
"#;

        let config = SynthConfig::from_yaml(yaml).unwrap();
        assert_eq!(config.name, "product_reviews");
        assert!(config.source.is_none());
        assert!(matches!(config.provider, ProviderConfig::Anthropic { .. }));
        assert!(matches!(config.generation.task, GenerationTask::Generate));
        assert_eq!(config.generation.categories.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_parse_validation_config() {
        let yaml = r#"
name: "with_validation"

provider:
  type: openai
  model: "gpt-4o-mini"

generation:
  task: generate
  count: 10
  template: "Generate JSON: {\"q\": \"...\", \"a\": \"...\"}"

output:
  format: jsonl
  path: "./output.jsonl"

validation:
  min_length: 20
  max_length: 1000
  json: true
  json_schema:
    - question
    - answer
  blocklist: true
  repetition: true
  dedupe: normalized
"#;

        let config = SynthConfig::from_yaml(yaml).unwrap();
        let v = config.validation.unwrap();
        assert_eq!(v.min_length, Some(20));
        assert_eq!(v.max_length, Some(1000));
        assert!(v.json);
        assert_eq!(v.json_schema.unwrap(), vec!["question", "answer"]);
        assert!(v.blocklist);
        assert!(v.repetition);
        assert!(matches!(v.dedupe, Some(DedupeStrategy::Normalized)));
    }
}
