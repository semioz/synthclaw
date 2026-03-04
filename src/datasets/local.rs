use polars::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use super::{DataSource, DatasetInfo, Record};
use crate::config::FileFormat;
use crate::{Error, Result};

pub struct LocalSource {
    path: PathBuf,
    format: FileFormat,
    info: DatasetInfo,
}

impl LocalSource {
    pub fn new(path: PathBuf, format: FileFormat) -> Result<Self> {
        if !path.exists() {
            return Err(Error::Dataset(format!("File not found: {:?}", path)));
        }

        let info = Self::detect_info(&path, &format)?;

        Ok(Self { path, format, info })
    }

    fn detect_info(path: &PathBuf, format: &FileFormat) -> Result<DatasetInfo> {
        let (columns, num_rows) = match format {
            FileFormat::Jsonl => Self::detect_jsonl_info(path)?,
            FileFormat::Json => Self::detect_json_info(path)?,
            FileFormat::Csv => Self::detect_csv_info(path)?,
            FileFormat::Parquet => Self::detect_parquet_info(path)?,
        };

        Ok(DatasetInfo {
            name: path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("local")
                .to_string(),
            description: None,
            num_rows,
            columns,
            splits: vec![],
        })
    }

    fn detect_jsonl_info(path: &PathBuf) -> Result<(Vec<String>, usize)> {
        let file = File::open(path).map_err(|e| Error::Dataset(e.to_string()))?;
        let reader = BufReader::new(file);
        let mut columns = Vec::new();
        let mut num_rows = 0;

        for (i, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| Error::Dataset(e.to_string()))?;
            if line.trim().is_empty() {
                continue;
            }
            num_rows += 1;

            if i == 0 {
                let obj: serde_json::Value =
                    serde_json::from_str(&line).map_err(|e| Error::Dataset(e.to_string()))?;
                if let Some(map) = obj.as_object() {
                    columns = map.keys().cloned().collect();
                }
            }
        }

        Ok((columns, num_rows))
    }

    fn detect_json_info(path: &PathBuf) -> Result<(Vec<String>, usize)> {
        let file = File::open(path).map_err(|e| Error::Dataset(e.to_string()))?;
        let data: serde_json::Value =
            serde_json::from_reader(file).map_err(|e| Error::Dataset(e.to_string()))?;

        match data {
            serde_json::Value::Array(arr) => {
                let num_rows = arr.len();
                let columns = arr
                    .first()
                    .and_then(|v| v.as_object())
                    .map(|m| m.keys().cloned().collect())
                    .unwrap_or_default();
                Ok((columns, num_rows))
            }
            _ => Err(Error::Dataset("JSON file must contain an array".into())),
        }
    }

    fn detect_csv_info(path: &PathBuf) -> Result<(Vec<String>, usize)> {
        let df = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some(path.clone()))
            .map_err(|e| Error::Dataset(e.to_string()))?
            .finish()
            .map_err(|e| Error::Dataset(e.to_string()))?;

        let columns: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        Ok((columns, df.height()))
    }

    fn detect_parquet_info(path: &PathBuf) -> Result<(Vec<String>, usize)> {
        let file = File::open(path).map_err(|e| Error::Dataset(e.to_string()))?;
        let df = ParquetReader::new(file)
            .finish()
            .map_err(|e| Error::Dataset(e.to_string()))?;

        let columns: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        Ok((columns, df.height()))
    }

    fn load_jsonl(&self, sample: Option<usize>) -> Result<Vec<Record>> {
        let file = File::open(&self.path).map_err(|e| Error::Dataset(e.to_string()))?;
        let reader = BufReader::new(file);
        let mut records = Vec::new();

        for (i, line) in reader.lines().enumerate() {
            if let Some(n) = sample {
                if records.len() >= n {
                    break;
                }
            }

            let line = line.map_err(|e| Error::Dataset(e.to_string()))?;
            if line.trim().is_empty() {
                continue;
            }

            let data: serde_json::Value =
                serde_json::from_str(&line).map_err(|e| Error::Dataset(e.to_string()))?;
            records.push(Record { data, index: i });
        }

        Ok(records)
    }

    fn load_json(&self, sample: Option<usize>) -> Result<Vec<Record>> {
        let file = File::open(&self.path).map_err(|e| Error::Dataset(e.to_string()))?;
        let data: serde_json::Value =
            serde_json::from_reader(file).map_err(|e| Error::Dataset(e.to_string()))?;

        match data {
            serde_json::Value::Array(arr) => {
                let limit = sample.unwrap_or(arr.len());
                Ok(arr
                    .into_iter()
                    .take(limit)
                    .enumerate()
                    .map(|(i, data)| Record { data, index: i })
                    .collect())
            }
            _ => Err(Error::Dataset("JSON file must contain an array".into())),
        }
    }

    fn load_csv(&self, sample: Option<usize>) -> Result<Vec<Record>> {
        let mut df = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some(self.path.clone()))
            .map_err(|e| Error::Dataset(e.to_string()))?
            .finish()
            .map_err(|e| Error::Dataset(e.to_string()))?;

        if let Some(n) = sample {
            df = df.head(Some(n));
        }

        dataframe_to_records(df)
    }

    fn load_parquet(&self, sample: Option<usize>) -> Result<Vec<Record>> {
        let file = File::open(&self.path).map_err(|e| Error::Dataset(e.to_string()))?;
        let mut df = ParquetReader::new(file)
            .finish()
            .map_err(|e| Error::Dataset(e.to_string()))?;

        if let Some(n) = sample {
            df = df.head(Some(n));
        }

        dataframe_to_records(df)
    }
}

impl DataSource for LocalSource {
    fn info(&self) -> &DatasetInfo {
        &self.info
    }

    fn load(&mut self, sample: Option<usize>) -> Result<Vec<Record>> {
        match self.format {
            FileFormat::Jsonl => self.load_jsonl(sample),
            FileFormat::Json => self.load_json(sample),
            FileFormat::Csv => self.load_csv(sample),
            FileFormat::Parquet => self.load_parquet(sample),
        }
    }
}

fn dataframe_to_records(df: DataFrame) -> Result<Vec<Record>> {
    let mut records = Vec::with_capacity(df.height());

    for i in 0..df.height() {
        let row = df
            .get(i)
            .ok_or_else(|| Error::Dataset("Row not found".into()))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_jsonl() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"text": "hello", "label": 1}}"#).unwrap();
        writeln!(file, r#"{{"text": "world", "label": 0}}"#).unwrap();
        writeln!(file, r#"{{"text": "test", "label": 1}}"#).unwrap();

        let mut source = LocalSource::new(file.path().to_path_buf(), FileFormat::Jsonl).unwrap();
        let records = source.load(Some(2)).unwrap();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].data["text"], "hello");
        assert_eq!(records[1].data["text"], "world");
    }

    #[test]
    fn test_load_json() {
        let mut file = NamedTempFile::new().unwrap();
        write!(
            file,
            r#"[{{"text": "a", "n": 1}}, {{"text": "b", "n": 2}}]"#
        )
        .unwrap();

        let mut source = LocalSource::new(file.path().to_path_buf(), FileFormat::Json).unwrap();
        let records = source.load(None).unwrap();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].data["text"], "a");
    }

    #[test]
    fn test_local_source_info() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"col1": "val1", "col2": 123}}"#).unwrap();
        writeln!(file, r#"{{"col1": "val2", "col2": 456}}"#).unwrap();

        let source = LocalSource::new(file.path().to_path_buf(), FileFormat::Jsonl).unwrap();
        let info = source.info();

        assert_eq!(info.num_rows, 2);
        assert!(info.columns.contains(&"col1".to_string()));
        assert!(info.columns.contains(&"col2".to_string()));
    }
}
