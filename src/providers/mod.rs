mod anthropic;
mod openai;

pub use anthropic::AnthropicProvider;
pub use openai::OpenAIProvider;

use async_trait::async_trait;
use crate::config::ProviderConfig;
use crate::Result;

#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub prompt: String,
    pub system_prompt: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct GenerationResponse {
    pub content: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct ModelPricing {
    pub input_per_million: f64,
    pub output_per_million: f64,
}

#[async_trait]
pub trait LLMProvider: Send + Sync {
    async fn generate(&self, request: GenerationRequest) -> Result<GenerationResponse>;
    fn estimate_cost(&self, input_tokens: u32, output_tokens: u32) -> f64;
    fn name(&self) -> &str;
    fn model(&self) -> &str;
}

pub fn create_provider(config: &ProviderConfig) -> Result<Box<dyn LLMProvider>> {
    match config {
        ProviderConfig::OpenAI {
            model,
            api_key,
            base_url,
            temperature,
            max_tokens,
        } => {
            let provider = OpenAIProvider::new(
                model.clone(),
                api_key.clone(),
                base_url.clone(),
                *temperature,
                *max_tokens,
            )?;
            Ok(Box::new(provider))
        }
        ProviderConfig::Anthropic {
            model,
            api_key,
            temperature,
            max_tokens,
        } => {
            let provider = AnthropicProvider::new(
                model.clone(),
                api_key.clone(),
                *temperature,
                *max_tokens,
            )?;
            Ok(Box::new(provider))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_openai_provider() {
        let config = ProviderConfig::OpenAI {
            model: "gpt-4o-mini".to_string(),
            api_key: Some("test-key".to_string()),
            base_url: None,
            temperature: Some(0.7),
            max_tokens: Some(1000),
        };

        let provider = create_provider(&config).unwrap();
        assert_eq!(provider.name(), "openai");
        assert_eq!(provider.model(), "gpt-4o-mini");
    }

    #[test]
    fn test_create_anthropic_provider() {
        let config = ProviderConfig::Anthropic {
            model: "claude-3-5-sonnet-20241022".to_string(),
            api_key: Some("test-key".to_string()),
            temperature: None,
            max_tokens: None,
        };

        let provider = create_provider(&config).unwrap();
        assert_eq!(provider.name(), "anthropic");
        assert_eq!(provider.model(), "claude-3-5-sonnet-20241022");
    }
}
