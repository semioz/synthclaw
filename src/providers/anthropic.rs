use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::{GenerationRequest, GenerationResponse, LLMProvider, ModelPricing};
use crate::{Error, Result};

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    model: String,
    default_temperature: Option<f32>,
    default_max_tokens: Option<u32>,
    pricing: ModelPricing,
}

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
    usage: Usage,
}

#[derive(Deserialize)]
struct ContentBlock {
    text: String,
}

#[derive(Deserialize)]
struct Usage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Deserialize)]
struct AnthropicError {
    error: ErrorDetail,
}

#[derive(Deserialize)]
struct ErrorDetail {
    message: String,
}

impl AnthropicProvider {
    pub fn new(
        model: String,
        api_key: Option<String>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> Result<Self> {
        let api_key = api_key
            .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
            .ok_or_else(|| {
                Error::Provider("ANTHROPIC_API_KEY not set and no api_key provided".into())
            })?;

        let client = Client::new();
        let pricing = get_model_pricing(&model);

        Ok(Self {
            client,
            api_key,
            model,
            default_temperature: temperature,
            default_max_tokens: max_tokens,
            pricing,
        })
    }
}

#[async_trait]
impl LLMProvider for AnthropicProvider {
    async fn generate(&self, request: GenerationRequest) -> Result<GenerationResponse> {
        let temperature = request.temperature.or(self.default_temperature);
        let max_tokens = request.max_tokens.or(self.default_max_tokens).unwrap_or(4096);

        let anthropic_request = AnthropicRequest {
            model: self.model.clone(),
            max_tokens,
            messages: vec![Message {
                role: "user".to_string(),
                content: request.prompt,
            }],
            system: request.system_prompt,
            temperature,
        };

        let response = self
            .client
            .post(ANTHROPIC_API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&anthropic_request)
            .send()
            .await
            .map_err(|e| Error::Provider(format!("Request failed: {}", e)))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| Error::Provider(format!("Failed to read response: {}", e)))?;

        if !status.is_success() {
            let error: AnthropicError = serde_json::from_str(&body)
                .map_err(|_| Error::Provider(format!("API error ({}): {}", status, body)))?;
            return Err(Error::Provider(format!(
                "Anthropic API error: {}",
                error.error.message
            )));
        }

        let response: AnthropicResponse = serde_json::from_str(&body)
            .map_err(|e| Error::Provider(format!("Failed to parse response: {}", e)))?;

        let content = response
            .content
            .first()
            .map(|c| c.text.clone())
            .ok_or_else(|| Error::Provider("No response content".into()))?;

        Ok(GenerationResponse {
            content,
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
        })
    }

    fn estimate_cost(&self, input_tokens: u32, output_tokens: u32) -> f64 {
        let input_cost = (input_tokens as f64 / 1_000_000.0) * self.pricing.input_per_million;
        let output_cost = (output_tokens as f64 / 1_000_000.0) * self.pricing.output_per_million;
        input_cost + output_cost
    }

    fn name(&self) -> &str {
        "anthropic"
    }

    fn model(&self) -> &str {
        &self.model
    }
}

fn get_model_pricing(model: &str) -> ModelPricing {
    match model {
        m if m.contains("claude-4-6-sonnet") || m.contains("claude-4.6-sonnet") => ModelPricing {
            input_per_million: 3.00,
            output_per_million: 15.00,
        },
        m if m.contains("claude-3-5-haiku") || m.contains("claude-3.5-haiku") => ModelPricing {
            input_per_million: 0.80,
            output_per_million: 4.00,
        },
        m if m.contains("claude-3-opus") => ModelPricing {
            input_per_million: 15.00,
            output_per_million: 75.00,
        },
        m if m.contains("claude-3-sonnet") => ModelPricing {
            input_per_million: 3.00,
            output_per_million: 15.00,
        },
        m if m.contains("claude-3-haiku") => ModelPricing {
            input_per_million: 0.25,
            output_per_million: 1.25,
        },
        _ => ModelPricing {
            input_per_million: 3.00,
            output_per_million: 15.00,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_pricing_sonnet() {
        let pricing = get_model_pricing("claude-3-5-sonnet-20241022");
        assert_eq!(pricing.input_per_million, 3.00);
        assert_eq!(pricing.output_per_million, 15.00);
    }

    #[test]
    fn test_model_pricing_haiku() {
        let pricing = get_model_pricing("claude-3-5-haiku-20241022");
        assert_eq!(pricing.input_per_million, 0.80);
        assert_eq!(pricing.output_per_million, 4.00);
    }

    #[test]
    fn test_model_pricing_opus() {
        let pricing = get_model_pricing("claude-3-opus-20240229");
        assert_eq!(pricing.input_per_million, 15.00);
        assert_eq!(pricing.output_per_million, 75.00);
    }

    #[test]
    fn test_cost_estimation() {
        let provider = AnthropicProvider::new(
            "claude-3-5-sonnet-20241022".to_string(),
            Some("test-key".to_string()),
            None,
            None,
        )
        .unwrap();

        // 1M input + 1M output tokens
        let cost = provider.estimate_cost(1_000_000, 1_000_000);
        assert!((cost - 18.00).abs() < 0.001); // $3 + $15 = $18
    }

    #[test]
    fn test_provider_name_and_model() {
        let provider = AnthropicProvider::new(
            "claude-3-5-sonnet-20241022".to_string(),
            Some("test-key".to_string()),
            Some(0.7),
            Some(2000),
        )
        .unwrap();

        assert_eq!(provider.name(), "anthropic");
        assert_eq!(provider.model(), "claude-3-5-sonnet-20241022");
    }

    #[test]
    fn test_missing_api_key_error() {
        unsafe { std::env::remove_var("ANTHROPIC_API_KEY") };
        let result = AnthropicProvider::new(
            "claude-3-5-sonnet-20241022".to_string(),
            None,
            None,
            None,
        );
        assert!(result.is_err());
    }
}
