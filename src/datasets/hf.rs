use polars::prelude::*;
use reqwest::blocking::Client;
use serde::Deserialize;
use std::io::Cursor;


// TODO: create skill file for the cli/lib so that agents can use it.

use super::{DatasetInfo, DataSource, Record};
use crate::{Error, Result};

const HF_DATASETS_SERVER: &str = "https://datasets-server.huggingface.co";

pub struct HuggingFaceSource {
    client: Client,
    dataset: String,
    subset: Option<String>,
    split: String,
    columns: Option<Vec<String>>,
    info: DatasetInfo,
    parquet_urls: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Split {
    pub name: String,
    pub num_rows: usize,
}

#[derive(Deserialize)]
struct InfoResponse {
    dataset_info: std::collections::HashMap<String, ConfigInfo>,
}

#[derive(Deserialize)]
struct ConfigInfo {
    description: Option<String>,
    features: std::collections::HashMap<String, serde_json::Value>,
    splits: std::collections::HashMap<String, SplitInfo>,
}

#[derive(Deserialize)]
struct SplitInfo {
    name: String,
    num_examples: usize,
}

#[derive(Deserialize)]
struct ParquetResponse {
    parquet_files: Vec<ParquetFile>,
}

#[derive(Deserialize)]
struct ParquetFile {
    config: String,
    split: String,
    url: String,
}

impl HuggingFaceSource {
    pub fn new(
        dataset: String,
        subset: Option<String>,
        split: String,
        columns: Option<Vec<String>>,
    ) -> Result<Self> {
        let client = Client::new();
        let mut source = Self {
            client,
            dataset,
            subset,
            split,
            columns,
            info: DatasetInfo::default(),
            parquet_urls: Vec::new(),
        };
        source.fetch_info()?;
        source.fetch_parquet_urls()?;
        Ok(source)
    }

    fn fetch_info(&mut self) -> Result<()> {
        let url = format!("{}/info?dataset={}", HF_DATASETS_SERVER, self.dataset);
        let response: InfoResponse = self
            .client
            .get(&url)
            .send()
            .map_err(|e| Error::Dataset(format!("Failed to fetch dataset info: {}", e)))?
            .json()
            .map_err(|e| Error::Dataset(format!("Failed to parse dataset info: {}", e)))?;

        let config_name = self.subset.as_deref().unwrap_or("default");
        let config = response.dataset_info.get(config_name).ok_or_else(|| {
            Error::Dataset(format!("Config '{}' not found in dataset", config_name))
        })?;

        let splits: Vec<Split> = config
            .splits
            .values()
            .map(|s| Split {
                name: s.name.clone(),
                num_rows: s.num_examples,
            })
            .collect();

        let split_info = splits
            .iter()
            .find(|s| s.name == self.split)
            .ok_or_else(|| Error::Dataset(format!("Split '{}' not found", self.split)))?;

        self.info = DatasetInfo {
            name: self.dataset.clone(),
            description: config.description.clone(),
            num_rows: split_info.num_rows,
            columns: config.features.keys().cloned().collect(),
            splits,
        };

        Ok(())
    }

    fn fetch_parquet_urls(&mut self) -> Result<()> {
        let url = format!("{}/parquet?dataset={}", HF_DATASETS_SERVER, self.dataset);
        let response: ParquetResponse = self
            .client
            .get(&url)
            .send()
            .map_err(|e| Error::Dataset(format!("Failed to fetch parquet URLs: {}", e)))?
            .json()
            .map_err(|e| Error::Dataset(format!("Failed to parse parquet response: {}", e)))?;

        let config_name = self.subset.as_deref().unwrap_or("default");
        self.parquet_urls = response
            .parquet_files
            .into_iter()
            .filter(|f| f.config == config_name && f.split == self.split)
            .map(|f| f.url)
            .collect();

        if self.parquet_urls.is_empty() {
            return Err(Error::Dataset(format!(
                "No parquet files found for config '{}' split '{}'",
                config_name, self.split
            )));
        }

        Ok(())
    }

    fn download_and_read_parquet(&self) -> Result<DataFrame> {
        let mut dfs = Vec::new();

        for url in &self.parquet_urls {
            let bytes = self
                .client
                .get(url)
                .send()
                .map_err(|e| Error::Dataset(format!("Failed to download parquet: {}", e)))?
                .bytes()
                .map_err(|e| Error::Dataset(format!("Failed to read parquet bytes: {}", e)))?;

            let cursor = Cursor::new(bytes.to_vec());
            let df = ParquetReader::new(cursor)
                .finish()
                .map_err(|e| Error::Dataset(format!("Failed to parse parquet: {}", e)))?;
            dfs.push(df);
        }

        if dfs.len() == 1 {
            Ok(dfs.remove(0))
        } else {
            let lazy_frames: Vec<_> = dfs.into_iter().map(|df| df.lazy()).collect();
            concat(&lazy_frames, UnionArgs::default())
                .map_err(|e| Error::Dataset(format!("Failed to concat dataframes: {}", e)))?
                .collect()
                .map_err(|e| Error::Dataset(format!("Failed to collect dataframe: {}", e)))
        }
    }
}

impl DataSource for HuggingFaceSource {
    fn info(&self) -> &DatasetInfo {
        &self.info
    }

    fn load(&mut self, sample: Option<usize>) -> Result<Vec<Record>> {
        let mut df = self.download_and_read_parquet()?;

        if let Some(cols) = &self.columns {
            let col_exprs: Vec<_> = cols.iter().map(|c| col(c)).collect();
            df = df
                .lazy()
                .select(col_exprs)
                .collect()
                .map_err(|e| Error::Dataset(format!("Failed to select columns: {}", e)))?;
        }

        if let Some(n) = sample {
            df = df.head(Some(n));
        }

        let mut records = Vec::with_capacity(df.height());
        for i in 0..df.height() {
            let row = df.get(i).ok_or_else(|| Error::Dataset("Row not found".into()))?;
            let mut map = serde_json::Map::new();

            for (col_name, value) in df.get_column_names().iter().zip(row.iter()) {
                let json_value = anyvalue_to_json(value);
                map.insert(col_name.to_string(), json_value);
            }

            records.push(Record {
                data: serde_json::Value::Object(map),
                index: i,
            });
        }

        Ok(records)
    }
}

fn anyvalue_to_json(value: &AnyValue) -> serde_json::Value {
    match value {
        AnyValue::Null => serde_json::Value::Null,
        AnyValue::Boolean(b) => serde_json::Value::Bool(*b),
        AnyValue::String(s) => serde_json::Value::String(s.to_string()),
        AnyValue::StringOwned(s) => serde_json::Value::String(s.to_string()),
        AnyValue::Float32(n) => serde_json::Number::from_f64(*n as f64)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        AnyValue::Float64(n) => serde_json::Number::from_f64(*n)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        other => serde_json::Value::String(format!("{}", other)),
    }
}

pub async fn search_datasets(query: &str, limit: usize) -> Result<Vec<DatasetSearchResult>> {
    let client = reqwest::Client::new();
    let url = format!(
        "https://huggingface.co/api/datasets?search={}&limit={}",
        query, limit
    );

    let results: Vec<DatasetSearchResult> = client
        .get(&url)
        .send()
        .await
        .map_err(|e| Error::Dataset(format!("Search failed: {}", e)))?
        .json()
        .await
        .map_err(|e| Error::Dataset(format!("Failed to parse search results: {}", e)))?;

    Ok(results)
}

#[derive(Debug, Deserialize)]
pub struct DatasetSearchResult {
    pub id: String,
    #[serde(default)]
    pub likes: u32,
    #[serde(default)]
    pub downloads: u64,
}

pub async fn get_dataset_info(dataset: &str) -> Result<DatasetInfo> {
    let client = reqwest::Client::new();
    let url = format!("{}/info?dataset={}", HF_DATASETS_SERVER, dataset);

    let response: InfoResponse = client
        .get(&url)
        .send()
        .await
        .map_err(|e| Error::Dataset(format!("Failed to fetch info: {}", e)))?
        .json()
        .await
        .map_err(|e| Error::Dataset(format!("Failed to parse info: {}", e)))?;

    let (_config_name, config) = response
        .dataset_info
        .into_iter()
        .next()
        .ok_or_else(|| Error::Dataset("No config found".into()))?;

    let splits: Vec<Split> = config
        .splits
        .values()
        .map(|s| Split {
            name: s.name.clone(),
            num_rows: s.num_examples,
        })
        .collect();

    let total_rows: usize = splits.iter().map(|s| s.num_rows).sum();

    Ok(DatasetInfo {
        name: dataset.to_string(),
        description: config.description,
        num_rows: total_rows,
        columns: config.features.keys().cloned().collect(),
        splits,
    })
}

pub async fn preview_dataset(
    dataset: &str,
    config: Option<&str>,
    split: &str,
    rows: usize,
) -> Result<Vec<serde_json::Value>> {
    let client = reqwest::Client::new();
    let config = config.unwrap_or("default");
    let url = format!(
        "{}/rows?dataset={}&config={}&split={}&offset=0&length={}",
        HF_DATASETS_SERVER, dataset, config, split, rows.min(100)
    );

    #[derive(Deserialize)]
    struct RowsResponse {
        rows: Vec<RowEntry>,
    }

    #[derive(Deserialize)]
    struct RowEntry {
        row: serde_json::Value,
    }

    let response: RowsResponse = client
        .get(&url)
        .send()
        .await
        .map_err(|e| Error::Dataset(format!("Failed to fetch rows: {}", e)))?
        .json()
        .await
        .map_err(|e| Error::Dataset(format!("Failed to parse rows: {}", e)))?;

    Ok(response.rows.into_iter().map(|r| r.row).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_search_datasets() {
        let results = search_datasets("sentiment", 5).await.unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[tokio::test]
    async fn test_get_dataset_info() {
        let info = get_dataset_info("cornell-movie-review-data/rotten_tomatoes")
            .await
            .unwrap();
        assert_eq!(info.name, "cornell-movie-review-data/rotten_tomatoes");
        assert!(info.columns.contains(&"text".to_string()));
        assert!(info.columns.contains(&"label".to_string()));
    }

    #[tokio::test]
    async fn test_preview_dataset() {
        let rows = preview_dataset(
            "cornell-movie-review-data/rotten_tomatoes",
            None,
            "train",
            3,
        )
        .await
        .unwrap();
        assert_eq!(rows.len(), 3);
        assert!(rows[0].get("text").is_some());
    }

    #[test]
    fn test_huggingface_source_load() {
        let mut source = HuggingFaceSource::new(
            "cornell-movie-review-data/rotten_tomatoes".to_string(),
            None,
            "train".to_string(),
            None,
        )
        .unwrap();

        let records = source.load(Some(10)).unwrap();
        assert_eq!(records.len(), 10);
        assert!(records[0].data.get("text").is_some());
    }
}
