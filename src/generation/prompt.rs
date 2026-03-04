use crate::datasets::Record;
use crate::{Error, Result};
use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;

pub struct PromptBuilder {
    template: String,
    system_prompt: String,
}

impl PromptBuilder {
    pub fn new(template: String, system_prompt: Option<String>, is_augment: bool) -> Self {
        let system_prompt = system_prompt.unwrap_or_else(|| {
            if is_augment {
                default_system_prompt_augment().to_string()
            } else {
                default_system_prompt_generate().to_string()
            }
        });
        Self {
            template,
            system_prompt,
        }
    }

    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    /// Build prompt for from-scratch generation with category
    pub fn build_for_category(&self, category: &str, index: usize) -> String {
        let mut vars = HashMap::new();
        vars.insert("category".to_string(), Value::String(category.to_string()));
        vars.insert("index".to_string(), Value::Number(index.into()));
        self.substitute(&vars)
    }

    /// Build prompt for augmenting existing data
    pub fn build_for_record(&self, record: &Record) -> String {
        let vars = self.extract_vars(&record.data);
        self.substitute(&vars)
    }

    /// Extract variable names from template (e.g., {category}, {text})
    pub fn required_variables(&self) -> Vec<String> {
        let re = Regex::new(r"\{(\w+)\}").unwrap();
        re.captures_iter(&self.template)
            .map(|cap| cap[1].to_string())
            .collect()
    }

    /// Validate that all required variables will be available
    pub fn validate_for_generate(&self, categories: &Option<Vec<String>>) -> Result<()> {
        let required = self.required_variables();

        for var in &required {
            match var.as_str() {
                "category" => {
                    if categories.is_none()
                        || categories.as_ref().map(|c| c.is_empty()).unwrap_or(true)
                    {
                        return Err(Error::Config(
                            "Template uses {category} but no categories provided".to_string(),
                        ));
                    }
                }
                "index" => {} // always available
                other => {
                    return Err(Error::Config(format!(
                        "Template uses {{{}}} which is not available in generate mode. Available: {{category}}, {{index}}",
                        other
                    )));
                }
            }
        }
        Ok(())
    }

    /// Validate that required variables exist in source data
    pub fn validate_for_augment(&self, available_columns: &[String]) -> Result<()> {
        let required = self.required_variables();

        for var in &required {
            if var != "index" && !available_columns.contains(var) {
                return Err(Error::Config(format!(
                    "Template uses {{{}}} but source data only has columns: {:?}",
                    var, available_columns
                )));
            }
        }
        Ok(())
    }

    fn substitute(&self, vars: &HashMap<String, Value>) -> String {
        let mut result = self.template.clone();
        for (key, value) in vars {
            let placeholder = format!("{{{}}}", key);
            let replacement = match value {
                Value::String(s) => s.clone(),
                Value::Number(n) => n.to_string(),
                Value::Bool(b) => b.to_string(),
                Value::Null => "null".to_string(),
                Value::Array(arr) => serde_json::to_string(arr).unwrap_or_default(),
                Value::Object(obj) => serde_json::to_string(obj).unwrap_or_default(),
            };
            result = result.replace(&placeholder, &replacement);
        }
        result
    }

    fn extract_vars(&self, data: &Value) -> HashMap<String, Value> {
        let mut vars = HashMap::new();
        if let Value::Object(map) = data {
            for (key, value) in map {
                vars.insert(key.clone(), value.clone());
            }
        }
        vars
    }
}

pub fn default_system_prompt_generate() -> &'static str {
    r#"You are a synthetic data generation assistant. Your task is to generate realistic, high-quality training data.

Rules:
- Output ONLY the requested content, nothing else
- No explanations, meta-commentary, or surrounding text
- No markdown formatting unless explicitly requested
- Generate diverse, realistic examples that could plausibly exist in the real world
- Vary your outputs - avoid repetitive patterns or templates
- Match the tone and style appropriate for the content type
- If generating text that would have a label (sentiment, category, etc.), make the content clearly match that label"#
}

pub fn default_system_prompt_augment() -> &'static str {
    r#"You are a data augmentation assistant. Your task is to transform input data while preserving its essential properties.

Rules:
- Output ONLY the transformed content, nothing else
- No explanations, meta-commentary, or surrounding text
- No markdown formatting unless explicitly requested
- Preserve the original meaning, sentiment, and intent
- If the data has a label (positive/negative, category, etc.), the augmented version must retain the same label
- Make meaningful changes - simple word swaps are not sufficient
- The output should be natural and fluent"#
}

pub fn default_template_for_generate() -> String {
    r#"Generate a realistic example of: {category}

Requirements:
- Authentic, natural language
- Specific details that make it believable
- 2-5 sentences unless otherwise specified
- Diverse - vary structure and content"#
        .to_string()
}

pub fn default_template_for_augment(strategy: &str) -> String {
    match strategy {
        "paraphrase" => {
            r#"Paraphrase the following text. Preserve the original meaning and sentiment exactly.

Input: {text}

Paraphrased version:"#
                .to_string()
        }

        "style_transfer" => {
            r#"Rewrite the following text in a different style while preserving the core meaning.

Input: {text}
Target style: {style}

Rewritten version:"#
                .to_string()
        }

        "back_translation" => {
            r#"Rephrase this text as if it was translated to another language and back. Keep the same meaning but use different word choices and sentence structures.

Input: {text}

Rephrased version:"#
                .to_string()
        }

        _ => {
            r#"Transform the following text while preserving its meaning:

Input: {text}

Transformed version:"#
                .to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_for_category() {
        let builder = PromptBuilder::new(
            "Generate a {category} review (item #{index})".to_string(),
            None,
            false,
        );
        let result = builder.build_for_category("electronics", 5);
        assert_eq!(result, "Generate a electronics review (item #5)");
    }

    #[test]
    fn test_build_for_record() {
        let builder =
            PromptBuilder::new("Paraphrase: {text}\nLabel: {label}".to_string(), None, true);
        let record = Record {
            data: serde_json::json!({
                "text": "This movie is great!",
                "label": 1
            }),
            index: 0,
        };
        let result = builder.build_for_record(&record);
        assert_eq!(result, "Paraphrase: This movie is great!\nLabel: 1");
    }

    #[test]
    fn test_required_variables() {
        let builder = PromptBuilder::new(
            "Hello {name}, your {item} for {category} is ready".to_string(),
            None,
            false,
        );
        let vars = builder.required_variables();
        assert!(vars.contains(&"name".to_string()));
        assert!(vars.contains(&"item".to_string()));
        assert!(vars.contains(&"category".to_string()));
    }

    #[test]
    fn test_validate_generate_missing_categories() {
        let builder = PromptBuilder::new("Generate a {category} example".to_string(), None, false);
        let result = builder.validate_for_generate(&None);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_generate_with_categories() {
        let builder = PromptBuilder::new("Generate a {category} example".to_string(), None, false);
        let result = builder.validate_for_generate(&Some(vec!["test".to_string()]));
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_augment_missing_column() {
        let builder =
            PromptBuilder::new("Paraphrase: {text} with {missing}".to_string(), None, true);
        let result = builder.validate_for_augment(&["text".to_string()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_augment_valid() {
        let builder = PromptBuilder::new("Paraphrase: {text}".to_string(), None, true);
        let result = builder.validate_for_augment(&["text".to_string(), "label".to_string()]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_default_system_prompts_exist() {
        assert!(!default_system_prompt_generate().is_empty());
        assert!(!default_system_prompt_augment().is_empty());
    }
}
