use crate::config::DedupeStrategy;
use crate::generation::GenerationResult;
use std::collections::HashSet;

pub enum Deduplicator {
    Exact,
    Normalized,
    Jaccard { n: usize, threshold: f32 },
}

impl Default for Deduplicator {
    fn default() -> Self {
        Self::Normalized
    }
}

impl From<&DedupeStrategy> for Deduplicator {
    fn from(s: &DedupeStrategy) -> Self {
        match s {
            DedupeStrategy::Exact => Self::Exact,
            DedupeStrategy::Normalized => Self::Normalized,
            DedupeStrategy::Jaccard => Self::Jaccard {
                n: 2,
                threshold: 0.7,
            },
        }
    }
}

impl Deduplicator {
    pub fn dedupe(&self, results: Vec<GenerationResult>) -> Vec<GenerationResult> {
        match self {
            Self::Exact => {
                let mut seen = HashSet::new();
                results
                    .into_iter()
                    .filter(|r| seen.insert(r.content.clone()))
                    .collect()
            }
            Self::Normalized => {
                let mut seen = HashSet::new();
                results
                    .into_iter()
                    .filter(|r| seen.insert(normalize(&r.content)))
                    .collect()
            }
            Self::Jaccard { n, threshold } => {
                let mut kept = vec![];
                let mut ngrams_list: Vec<HashSet<String>> = vec![];

                for r in results {
                    let ng = ngrams(&r.content, *n);
                    if !ngrams_list
                        .iter()
                        .any(|existing| jaccard(&ng, existing) >= *threshold)
                    {
                        ngrams_list.push(ng);
                        kept.push(r);
                    }
                }
                kept
            }
        }
    }
}

fn normalize(s: &str) -> String {
    s.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn ngrams(s: &str, n: usize) -> HashSet<String> {
    let words: Vec<_> = s.split_whitespace().collect();
    if words.len() < n {
        return HashSet::from([s.to_lowercase()]);
    }
    words
        .windows(n)
        .map(|w| w.join(" ").to_lowercase())
        .collect()
}

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    a.intersection(b).count() as f32 / a.union(b).count() as f32
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
    fn test_exact() {
        let d = Deduplicator::Exact;
        let res = d.dedupe(vec![r("a"), r("a"), r("b")]);
        assert_eq!(res.len(), 2);
    }

    #[test]
    fn test_normalized() {
        let d = Deduplicator::Normalized;
        let res = d.dedupe(vec![r("Hello"), r("  hello  "), r("HELLO"), r("other")]);
        assert_eq!(res.len(), 2);
    }

    #[test]
    fn test_jaccard() {
        let d = Deduplicator::Jaccard {
            n: 2,
            threshold: 0.5,
        };
        let res = d.dedupe(vec![
            r("the quick brown fox jumps over the lazy dog"),
            r("the quick brown fox jumps over the lazy cat"),
            r("completely different text here"),
        ]);
        assert_eq!(res.len(), 2);
    }
}
