pub mod hf;
mod local;

pub use hf::{HuggingFaceSource, Split};
pub use local::LocalSource;

use crate::Result;
use serde_json::Value;

pub struct Record {
    pub data: Value,
    pub index: usize,
}

pub trait DataSource: Send + Sync {
    fn info(&self) -> &DatasetInfo;
    fn load(&mut self, sample: Option<usize>) -> Result<Vec<Record>>;
}

#[derive(Default)]
pub struct DatasetInfo {
    pub name: String,
    pub description: Option<String>,
    pub num_rows: usize,
    pub columns: Vec<String>,
    pub splits: Vec<Split>,
}
