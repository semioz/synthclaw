mod dedup;
mod validators;

pub use dedup::*;
pub use validators::*;

use crate::config::ValidationConfig;
use crate::generation::GenerationResult;

#[derive(Debug, Clone, Default)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
}

impl ValidationResult {
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            errors: vec![],
        }
    }

    pub fn invalid(error: impl Into<String>) -> Self {
        Self {
            is_valid: false,
            errors: vec![error.into()],
        }
    }

    pub fn merge(&mut self, other: Self) {
        if !other.is_valid {
            self.is_valid = false;
            self.errors.extend(other.errors);
        }
    }
}

pub trait Validator: Send + Sync {
    fn validate(&self, result: &GenerationResult) -> ValidationResult;
}

#[derive(Default)]
pub struct ValidationPipeline {
    validators: Vec<Box<dyn Validator>>,
}

impl ValidationPipeline {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_config(config: &ValidationConfig) -> Self {
        let mut p = Self::new();
        if let Some(n) = config.min_length {
            p = p.add(MinLength(n));
        }
        if let Some(n) = config.max_length {
            p = p.add(MaxLength(n));
        }
        if config.json {
            p = p.add(Json);
        }
        if let Some(fields) = &config.json_schema {
            let fields: Vec<&str> = fields.iter().map(|s| s.as_str()).collect();
            p = p.add(JsonSchema::require(&fields));
        }
        if config.blocklist {
            p = p.add(Blocklist::llm_artifacts());
        }
        if config.repetition {
            p = p.add(Repetition::default());
        }
        p
    }

    pub fn add<V: Validator + 'static>(mut self, v: V) -> Self {
        self.validators.push(Box::new(v));
        self
    }

    pub fn validate(&self, result: &GenerationResult) -> ValidationResult {
        self.validators
            .iter()
            .fold(ValidationResult::valid(), |mut acc, v| {
                acc.merge(v.validate(result));
                acc
            })
    }

    pub fn filter(
        &self,
        results: Vec<GenerationResult>,
    ) -> (
        Vec<GenerationResult>,
        Vec<(GenerationResult, ValidationResult)>,
    ) {
        let (mut valid, mut invalid) = (vec![], vec![]);
        for r in results {
            let v = self.validate(&r);
            if v.is_valid {
                valid.push(r);
            } else {
                invalid.push((r, v));
            }
        }
        (valid, invalid)
    }
}

#[derive(Debug, Clone, Default)]
pub struct ValidationStats {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub duplicates_removed: usize,
}

pub struct ValidatedResults {
    pub results: Vec<GenerationResult>,
    pub stats: ValidationStats,
    pub rejected: Vec<(GenerationResult, ValidationResult)>,
}

pub fn validate_and_dedupe(
    results: Vec<GenerationResult>,
    pipeline: &ValidationPipeline,
    dedup: Option<&Deduplicator>,
) -> ValidatedResults {
    let total = results.len();
    let (valid, rejected) = pipeline.filter(results);
    let failed = rejected.len();

    let (results, duplicates_removed) = match dedup {
        Some(d) => {
            let before = valid.len();
            let deduped = d.dedupe(valid);
            let removed = before - deduped.len();
            (deduped, removed)
        }
        None => (valid, 0),
    };

    ValidatedResults {
        stats: ValidationStats {
            total,
            passed: results.len(),
            failed,
            duplicates_removed,
        },
        results,
        rejected,
    }
}
