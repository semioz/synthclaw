use crate::config::{GenerationConfig, GenerationStrategy, GenerationTask, ProviderConfig, SourceConfig, SynthConfig};
use crate::datasets::{DataSource, HuggingFaceSource, LocalSource, Record};
use crate::providers::{create_provider, GenerationRequest, GenerationResponse, LLMProvider};
use crate::{Error, Result};

use super::prompt::{default_template_for_augment, default_template_for_generate, PromptBuilder};

use futures::stream::{self, StreamExt};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Default)]
pub struct GenerationStats {
    pub completed: AtomicUsize,
    pub failed: AtomicUsize,
    pub total_input_tokens: AtomicU64,
    pub total_output_tokens: AtomicU64,
}

impl GenerationStats {
    pub fn record_success(&self, response: &GenerationResponse) {
        self.completed.fetch_add(1, Ordering::Relaxed);
        self.total_input_tokens.fetch_add(response.input_tokens as u64, Ordering::Relaxed);
        self.total_output_tokens.fetch_add(response.output_tokens as u64, Ordering::Relaxed);
    }

    pub fn record_failure(&self) {
        self.failed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> StatsSnapshot {
        StatsSnapshot {
            completed: self.completed.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
            total_input_tokens: self.total_input_tokens.load(Ordering::Relaxed),
            total_output_tokens: self.total_output_tokens.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StatsSnapshot {
    pub completed: usize,
    pub failed: usize,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
}

#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub content: String,
    pub source_index: Option<usize>,
    pub category: Option<String>,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl GenerationResult {
    /// Parse content as JSON, extracting from markdown code blocks if needed
    pub fn parse_json(&self) -> Result<serde_json::Value> {
        let content = self.extract_json_content();
        serde_json::from_str(&content).map_err(|e| Error::Json(e))
    }

    /// Parse content as a typed JSON object
    pub fn parse_json_as<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
        let content = self.extract_json_content();
        serde_json::from_str(&content).map_err(|e| Error::Json(e))
    }

    fn extract_json_content(&self) -> String {
        let content = self.content.trim();

        // Try ```json blocks
        if let Some(start) = content.find("```json") {
            if let Some(end) = content[start + 7..].find("```") {
                return content[start + 7..start + 7 + end].trim().to_string();
            }
        }

        // Try generic ``` blocks
        if let Some(start) = content.find("```") {
            if let Some(end) = content[start + 3..].find("```") {
                let inner = content[start + 3..start + 3 + end].trim();
                if let Some(newline) = inner.find('\n') {
                    return inner[newline + 1..].trim().to_string();
                }
                return inner.to_string();
            }
        }

        content.to_string()
    }
}

struct GenerationTask_ {
    prompt: String,
    system_prompt: Option<String>,
    source_index: Option<usize>,
    category: Option<String>,
}

pub struct GenerationEngine {
    provider: Box<dyn LLMProvider>,
    config: GenerationConfig,
    stats: Arc<GenerationStats>,
}

impl GenerationEngine {
    pub fn new(provider_config: &ProviderConfig, generation_config: GenerationConfig) -> Result<Self> {
        let provider = create_provider(provider_config)?;
        Ok(Self {
            provider,
            config: generation_config,
            stats: Arc::new(GenerationStats::default()),
        })
    }

    pub fn stats(&self) -> Arc<GenerationStats> {
        Arc::clone(&self.stats)
    }

    pub fn provider(&self) -> &dyn LLMProvider {
        self.provider.as_ref()
    }

    /// Run generation and collect all results
    pub async fn run(&self, config: &SynthConfig) -> Result<Vec<GenerationResult>> {
        let tasks = self.build_tasks(config).await?;
        let results = Arc::new(Mutex::new(Vec::with_capacity(tasks.len())));
        
        stream::iter(tasks)
            .map(|task| {
                let provider = &self.provider;
                let stats = Arc::clone(&self.stats);
                let results = Arc::clone(&results);
                async move {
                    match self.execute_task(provider.as_ref(), task).await {
                        Ok(result) => {
                            stats.record_success(&GenerationResponse {
                                content: result.content.clone(),
                                input_tokens: result.input_tokens,
                                output_tokens: result.output_tokens,
                            });
                            results.lock().await.push(result);
                        }
                        Err(e) => {
                            stats.record_failure();
                            tracing::warn!("Generation failed: {}", e);
                        }
                    }
                }
            })
            .buffer_unordered(self.config.concurrency)
            .collect::<Vec<_>>()
            .await;

        let results = Arc::try_unwrap(results)
            .map_err(|_| Error::Provider("Failed to unwrap results".to_string()))?
            .into_inner();
        
        Ok(results)
    }

    /// Run generation with a callback for each result (for streaming output)
    pub async fn run_with_callback<F>(&self, config: &SynthConfig, on_result: F) -> Result<()>
    where
        F: FnMut(GenerationResult) + Send,
    {
        let tasks = self.build_tasks(config).await?;
        let callback = Arc::new(Mutex::new(on_result));
        
        stream::iter(tasks)
            .map(|task| {
                let provider = &self.provider;
                let stats = Arc::clone(&self.stats);
                let callback = Arc::clone(&callback);
                async move {
                    match self.execute_task(provider.as_ref(), task).await {
                        Ok(result) => {
                            stats.record_success(&GenerationResponse {
                                content: result.content.clone(),
                                input_tokens: result.input_tokens,
                                output_tokens: result.output_tokens,
                            });
                            callback.lock().await(result);
                        }
                        Err(e) => {
                            stats.record_failure();
                            tracing::warn!("Generation failed: {}", e);
                        }
                    }
                }
            })
            .buffer_unordered(self.config.concurrency)
            .collect::<Vec<_>>()
            .await;
        
        Ok(())
    }

    async fn build_tasks(&self, config: &SynthConfig) -> Result<Vec<GenerationTask_>> {
        let prompt_builder = self.create_prompt_builder();
        
        match &config.generation.task {
            GenerationTask::Generate => self.build_generate_tasks(&prompt_builder),
            GenerationTask::Augment => self.build_augment_tasks(config, &prompt_builder).await,
        }
    }

    fn build_generate_tasks(&self, prompt_builder: &PromptBuilder) -> Result<Vec<GenerationTask_>> {
        let categories = self.config.categories.as_ref();
        let count = self.config.count;
        let system_prompt = Some(prompt_builder.system_prompt().to_string());

        let mut tasks = Vec::with_capacity(count);

        if let Some(cats) = categories {
            let per_category = count / cats.len();
            let remainder = count % cats.len();

            for (cat_idx, category) in cats.iter().enumerate() {
                let cat_count = per_category + if cat_idx < remainder { 1 } else { 0 };
                for i in 0..cat_count {
                    tasks.push(GenerationTask_ {
                        prompt: prompt_builder.build_for_category(category, i),
                        system_prompt: system_prompt.clone(),
                        source_index: None,
                        category: Some(category.clone()),
                    });
                }
            }
        } else {
            for i in 0..count {
                tasks.push(GenerationTask_ {
                    prompt: prompt_builder.build_for_category("default", i),
                    system_prompt: system_prompt.clone(),
                    source_index: None,
                    category: None,
                });
            }
        }

        Ok(tasks)
    }

    async fn build_augment_tasks(&self, config: &SynthConfig, prompt_builder: &PromptBuilder) -> Result<Vec<GenerationTask_>> {
        let source_config = config.source.as_ref()
            .ok_or_else(|| Error::Config("Augment task requires a source configuration".to_string()))?;

        let records = self.load_source_data(source_config.clone()).await?;
        let count_per = self.config.count_per_example.unwrap_or(1);
        let system_prompt = Some(prompt_builder.system_prompt().to_string());

        let mut tasks = Vec::with_capacity(records.len() * count_per);

        for record in &records {
            for _ in 0..count_per {
                tasks.push(GenerationTask_ {
                    prompt: prompt_builder.build_for_record(record),
                    system_prompt: system_prompt.clone(),
                    source_index: Some(record.index),
                    category: None,
                });
            }
        }

        Ok(tasks)
    }

    async fn load_source_data(&self, source_config: SourceConfig) -> Result<Vec<Record>> {
        // Run blocking IO operations in a separate thread pool
        tokio::task::spawn_blocking(move || {
            match source_config {
                SourceConfig::HuggingFace { dataset, subset, split, sample, columns } => {
                    let mut source = HuggingFaceSource::new(
                        dataset,
                        subset,
                        split,
                        columns,
                    )?;
                    source.load(sample)
                }
                SourceConfig::Local { path, format, sample } => {
                    let mut source = LocalSource::new(path, format)?;
                    source.load(sample)
                }
            }
        })
        .await
        .map_err(|e| Error::Dataset(format!("Task join error: {}", e)))?
    }

    fn create_prompt_builder(&self) -> PromptBuilder {
        let is_augment = matches!(&self.config.task, GenerationTask::Augment);
        
        let template = self.config.template.clone().unwrap_or_else(|| {
            match &self.config.task {
                GenerationTask::Generate => default_template_for_generate(),
                GenerationTask::Augment => {
                    let strategy = self.config.strategy.as_ref()
                        .map(|s| match s {
                            GenerationStrategy::Paraphrase => "paraphrase",
                            GenerationStrategy::StyleTransfer => "style_transfer",
                            GenerationStrategy::BackTranslation => "back_translation",
                            GenerationStrategy::Custom => "custom",
                        })
                        .unwrap_or("paraphrase");
                    default_template_for_augment(strategy)
                }
            }
        });

        PromptBuilder::new(template, self.config.system_prompt.clone(), is_augment)
    }

    async fn execute_task(&self, provider: &dyn LLMProvider, task: GenerationTask_) -> Result<GenerationResult> {
        let request = GenerationRequest {
            prompt: task.prompt,
            system_prompt: task.system_prompt,
            temperature: None,
            max_tokens: None,
        };

        let response = provider.generate(request).await?;

        Ok(GenerationResult {
            content: response.content,
            source_index: task.source_index,
            category: task.category,
            input_tokens: response.input_tokens,
            output_tokens: response.output_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;

    fn test_config() -> SynthConfig {
        SynthConfig {
            name: "test".to_string(),
            source: None,
            provider: ProviderConfig::OpenAI {
                model: "gpt-4o-mini".to_string(),
                api_key: Some("test-key".to_string()),
                base_url: None,
                temperature: None,
                max_tokens: None,
            },
            generation: GenerationConfig {
                task: GenerationTask::Generate,
                count: 10,
                count_per_example: None,
                concurrency: 2,
                strategy: None,
                strategy_config: Default::default(),
                template: Some("Generate a {category} example".to_string()),
                system_prompt: None,
                categories: Some(vec!["A".to_string(), "B".to_string()]),
            },
            output: OutputConfig {
                format: OutputFormat::Jsonl,
                path: "./output.jsonl".into(),
                batch_size: 100,
            },
            validation: None,
        }
    }

    #[test]
    fn test_build_generate_tasks() {
        let config = test_config();
        let engine = GenerationEngine::new(&config.provider, config.generation.clone()).unwrap();
        let prompt_builder = engine.create_prompt_builder();
        
        let tasks = engine.build_generate_tasks(&prompt_builder).unwrap();
        
        assert_eq!(tasks.len(), 10);
        // 5 for category A, 5 for category B
        let a_count = tasks.iter().filter(|t| t.category.as_deref() == Some("A")).count();
        let b_count = tasks.iter().filter(|t| t.category.as_deref() == Some("B")).count();
        assert_eq!(a_count, 5);
        assert_eq!(b_count, 5);
    }

    #[test]
    fn test_stats_tracking() {
        let stats = GenerationStats::default();
        
        stats.record_success(&GenerationResponse {
            content: "test".to_string(),
            input_tokens: 100,
            output_tokens: 50,
        });
        stats.record_success(&GenerationResponse {
            content: "test".to_string(),
            input_tokens: 200,
            output_tokens: 100,
        });
        stats.record_failure();

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.completed, 2);
        assert_eq!(snapshot.failed, 1);
        assert_eq!(snapshot.total_input_tokens, 300);
        assert_eq!(snapshot.total_output_tokens, 150);
    }
}
