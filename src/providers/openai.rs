use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::{GenerationRequest, GenerationResponse, LLMProvider, ModelPricing};
use crate::{Error, Result};

const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";

pub struct OpenAIProvider {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
    default_temperature: Option<f32>,
    default_max_tokens: Option<u32>,
    pricing: ModelPricing,
}

#[derive(Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OpenAIResponse {
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: String,
}

#[derive(Deserialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Deserialize)]
struct OpenAIError {
    error: ErrorDetail,
}

#[derive(Deserialize)]
struct ErrorDetail {
    message: String,
}

impl OpenAIProvider {
    pub fn new(
        model: String,
        api_key: Option<String>,
        base_url: Option<String>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> Result<Self> {
        let api_key = api_key
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .ok_or_else(|| {
                Error::Provider("OPENAI_API_KEY not set and no api_key provided".into())
            })?;

        let base_url = base_url.unwrap_or_else(|| OPENAI_API_URL.to_string());
        let client = Client::new();
        let pricing = get_model_pricing(&model);

        Ok(Self {
            client,
            api_key,
            base_url,
            model,
            default_temperature: temperature,
            default_max_tokens: max_tokens,
            pricing,
        })
    }
}

#[async_trait]
impl LLMProvider for OpenAIProvider {
    async fn generate(&self, request: GenerationRequest) -> Result<GenerationResponse> {
        let mut messages = Vec::new();

        if let Some(system) = &request.system_prompt {
            messages.push(Message {
                role: "system".to_string(),
                content: system.clone(),
            });
        }

        messages.push(Message {
            role: "user".to_string(),
            content: request.prompt,
        });

        let temperature = request.temperature.or(self.default_temperature);
        let max_tokens = request.max_tokens.or(self.default_max_tokens);

        let openai_request = OpenAIRequest {
            model: self.model.clone(),
            messages,
            temperature,
            max_tokens,
        };

        let response = self
            .client
            .post(&self.base_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&openai_request)
            .send()
            .await
            .map_err(|e| Error::Provider(format!("Request failed: {}", e)))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| Error::Provider(format!("Failed to read response: {}", e)))?;

        if !status.is_success() {
            let error: OpenAIError = serde_json::from_str(&body)
                .map_err(|_| Error::Provider(format!("API error ({}): {}", status, body)))?;
            return Err(Error::Provider(format!(
                "OpenAI API error: {}",
                error.error.message
            )));
        }

        let response: OpenAIResponse = serde_json::from_str(&body)
            .map_err(|e| Error::Provider(format!("Failed to parse response: {}", e)))?;

        let content = response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| Error::Provider("No response content".into()))?;

        Ok(GenerationResponse {
            content,
            input_tokens: response.usage.prompt_tokens,
            output_tokens: response.usage.completion_tokens,
        })
    }

    fn estimate_cost(&self, input_tokens: u32, output_tokens: u32) -> f64 {
        let input_cost = (input_tokens as f64 / 1_000_000.0) * self.pricing.input_per_million;
        let output_cost = (output_tokens as f64 / 1_000_000.0) * self.pricing.output_per_million;
        input_cost + output_cost
    }

    fn name(&self) -> &str {
        "openai"
    }

    fn model(&self) -> &str {
        &self.model
    }
}

fn get_model_pricing(model: &str) -> ModelPricing {
    match model {
        "gpt-4o" => ModelPricing {
            input_per_million: 2.50,
            output_per_million: 10.00,
        },
        "gpt-4o-mini" => ModelPricing {
            input_per_million: 0.15,
            output_per_million: 0.60,
        },
        "gpt-4-turbo" | "gpt-4-turbo-preview" => ModelPricing {
            input_per_million: 10.00,
            output_per_million: 30.00,
        },
        "gpt-4" => ModelPricing {
            input_per_million: 30.00,
            output_per_million: 60.00,
        },
        "gpt-3.5-turbo" => ModelPricing {
            input_per_million: 0.50,
            output_per_million: 1.50,
        },
        "o1" => ModelPricing {
            input_per_million: 15.00,
            output_per_million: 60.00,
        },
        "o1-mini" => ModelPricing {
            input_per_million: 3.00,
            output_per_million: 12.00,
        },
        "o3-mini" => ModelPricing {
            input_per_million: 1.10,
            output_per_million: 4.40,
        },
        _ => ModelPricing {
            input_per_million: 2.50,
            output_per_million: 10.00,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_pricing_gpt4o_mini() {
        let pricing = get_model_pricing("gpt-4o-mini");
        assert_eq!(pricing.input_per_million, 0.15);
        assert_eq!(pricing.output_per_million, 0.60);
    }

    #[test]
    fn test_cost_estimation() {
        let provider = OpenAIProvider::new(
            "gpt-4o-mini".to_string(),
            Some("test-key".to_string()),
            None,
            None,
            None,
        )
        .unwrap();

        // 1M input tokens + 1M output tokens
        let cost = provider.estimate_cost(1_000_000, 1_000_000);
        assert!((cost - 0.75).abs() < 0.001); // $0.15 + $0.60 = $0.75
    }

    #[test]
    fn test_provider_name_and_model() {
        let provider = OpenAIProvider::new(
            "gpt-4o".to_string(),
            Some("test-key".to_string()),
            None,
            Some(0.7),
            Some(1000),
        )
        .unwrap();

        assert_eq!(provider.name(), "openai");
        assert_eq!(provider.model(), "gpt-4o");
    }

    #[test]
    fn test_missing_api_key_error() {
        unsafe { std::env::remove_var("OPENAI_API_KEY") };
        let result = OpenAIProvider::new("gpt-4o".to_string(), None, None, None, None);
        assert!(result.is_err());
    }
}
