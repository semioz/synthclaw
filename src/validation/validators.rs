use super::{ValidationResult, Validator};
use crate::generation::GenerationResult;
use regex::Regex;
use std::collections::HashMap;

pub struct MinLength(pub usize);
pub struct MaxLength(pub usize);

impl Validator for MinLength {
    fn validate(&self, r: &GenerationResult) -> ValidationResult {
        if r.content.trim().len() >= self.0 {
            ValidationResult::valid()
        } else {
            ValidationResult::invalid(format!(
                "too short: {} < {}",
                r.content.trim().len(),
                self.0
            ))
        }
    }
}

impl Validator for MaxLength {
    fn validate(&self, r: &GenerationResult) -> ValidationResult {
        if r.content.len() <= self.0 {
            ValidationResult::valid()
        } else {
            ValidationResult::invalid(format!("too long: {} > {}", r.content.len(), self.0))
        }
    }
}

pub struct Json;
pub struct JsonSchema {
    required: Vec<String>,
}

impl JsonSchema {
    pub fn require(fields: &[&str]) -> Self {
        Self {
            required: fields.iter().map(|s| s.to_string()).collect(),
        }
    }
}

impl Validator for Json {
    fn validate(&self, r: &GenerationResult) -> ValidationResult {
        match serde_json::from_str::<serde_json::Value>(&extract_json(&r.content)) {
            Ok(_) => ValidationResult::valid(),
            Err(e) => ValidationResult::invalid(format!("invalid json: {}", e)),
        }
    }
}

impl Validator for JsonSchema {
    fn validate(&self, r: &GenerationResult) -> ValidationResult {
        let v: serde_json::Value = match serde_json::from_str(&extract_json(&r.content)) {
            Ok(v) => v,
            Err(e) => return ValidationResult::invalid(format!("invalid json: {}", e)),
        };

        let obj = match v.as_object() {
            Some(o) => o,
            None => return ValidationResult::invalid("expected json object"),
        };

        let missing: Vec<_> = self
            .required
            .iter()
            .filter(|f| !obj.contains_key(*f))
            .collect();
        if missing.is_empty() {
            ValidationResult::valid()
        } else {
            ValidationResult::invalid(format!("missing fields: {:?}", missing))
        }
    }
}

pub struct Blocklist(Vec<(Regex, &'static str)>);

impl Blocklist {
    pub fn llm_artifacts() -> Self {
        let patterns = [
            (r"(?i)^(sure|certainly|of course)[,!]?\s", "filler"),
            (r"(?i)^here('s| is)", "here is"),
            (r"(?i)^I('d| would) be happy to", "politeness"),
            (r"(?i)as an AI", "ai mention"),
            (r"(?i)I cannot|I can't|I'm unable", "refusal"),
        ];
        Self(
            patterns
                .iter()
                .filter_map(|(p, r)| Some((Regex::new(p).ok()?, *r)))
                .collect(),
        )
    }
}

impl Validator for Blocklist {
    fn validate(&self, r: &GenerationResult) -> ValidationResult {
        for (re, reason) in &self.0 {
            if re.is_match(&r.content) {
                return ValidationResult::invalid(format!("blocked: {}", reason));
            }
        }
        ValidationResult::valid()
    }
}

pub struct Repetition {
    pub max_ratio: f32,
    pub ngram_size: usize,
}

impl Default for Repetition {
    fn default() -> Self {
        Self {
            max_ratio: 0.5,
            ngram_size: 3,
        }
    }
}

impl Validator for Repetition {
    fn validate(&self, r: &GenerationResult) -> ValidationResult {
        let words: Vec<_> = r.content.split_whitespace().collect();
        if words.len() < self.ngram_size * 2 {
            return ValidationResult::valid();
        }

        let mut counts: HashMap<String, usize> = HashMap::new();
        for w in words.windows(self.ngram_size) {
            *counts.entry(w.join(" ").to_lowercase()).or_default() += 1;
        }

        let total = words.len() - self.ngram_size + 1;
        let repeated: usize = counts.values().filter(|&&c| c > 1).map(|c| c - 1).sum();
        let ratio = repeated as f32 / total as f32;

        if ratio <= self.max_ratio {
            ValidationResult::valid()
        } else {
            ValidationResult::invalid(format!(
                "repetitive: {:.0}% > {:.0}%",
                ratio * 100.0,
                self.max_ratio * 100.0
            ))
        }
    }
}

pub struct Custom<F>(pub F);

impl<F: Fn(&GenerationResult) -> ValidationResult + Send + Sync> Validator for Custom<F> {
    fn validate(&self, r: &GenerationResult) -> ValidationResult {
        self.0(r)
    }
}

fn extract_json(content: &str) -> String {
    let s = content.trim();
    if let Some(start) = s.find("```json") {
        if let Some(end) = s[start + 7..].find("```") {
            return s[start + 7..start + 7 + end].trim().to_string();
        }
    }
    if let Some(start) = s.find("```") {
        if let Some(end) = s[start + 3..].find("```") {
            let inner = s[start + 3..start + 3 + end].trim();
            return inner.lines().skip(1).collect::<Vec<_>>().join("\n");
        }
    }
    s.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(s: &str) -> GenerationResult {
        GenerationResult {
            content: s.to_string(),
            source_index: None,
            category: None,
            input_tokens: 0,
            output_tokens: 0,
        }
    }

    #[test]
    fn test_length() {
        assert!(!MinLength(10).validate(&r("short")).is_valid);
        assert!(MinLength(5).validate(&r("hello")).is_valid);
        assert!(MaxLength(10).validate(&r("short")).is_valid);
        assert!(!MaxLength(5).validate(&r("too long")).is_valid);
    }

    #[test]
    fn test_json() {
        assert!(Json.validate(&r(r#"{"a":1}"#)).is_valid);
        assert!(!Json.validate(&r("not json")).is_valid);
        assert!(Json.validate(&r("```json\n{\"a\":1}\n```")).is_valid);
    }

    #[test]
    fn test_schema() {
        let v = JsonSchema::require(&["a", "b"]);
        assert!(v.validate(&r(r#"{"a":1,"b":2}"#)).is_valid);
        assert!(!v.validate(&r(r#"{"a":1}"#)).is_valid);
    }

    #[test]
    fn test_blocklist() {
        let v = Blocklist::llm_artifacts();
        assert!(!v.validate(&r("Sure! Here you go")).is_valid);
        assert!(!v.validate(&r("As an AI, I")).is_valid);
        assert!(v.validate(&r("Normal text")).is_valid);
    }

    #[test]
    fn test_repetition() {
        let v = Repetition {
            max_ratio: 0.3,
            ngram_size: 2,
        };
        assert!(!v.validate(&r("the cat the cat the cat the cat")).is_valid);
        assert!(v.validate(&r("the quick brown fox jumps")).is_valid);
    }
}
