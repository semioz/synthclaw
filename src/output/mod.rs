use crate::generation::GenerationResult;
use crate::{Error, Result};
use polars::prelude::*;
use serde_json::json;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

pub trait OutputWriter: Send {
    fn write(&mut self, result: &GenerationResult) -> Result<()>;
    fn flush(&mut self) -> Result<()>;
}

pub struct JsonlWriter {
    writer: BufWriter<File>,
}

impl JsonlWriter {
    pub fn new(path: PathBuf) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }
}

impl OutputWriter for JsonlWriter {
    fn write(&mut self, result: &GenerationResult) -> Result<()> {
        let mut obj = json!({
            "content": result.content,
        });

        if let Some(idx) = result.source_index {
            obj["source_index"] = json!(idx);
        }
        if let Some(cat) = &result.category {
            obj["category"] = json!(cat);
        }

        obj["input_tokens"] = json!(result.input_tokens);
        obj["output_tokens"] = json!(result.output_tokens);

        writeln!(self.writer, "{}", serde_json::to_string(&obj)?)?;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

pub struct CsvWriter {
    writer: csv::Writer<File>,
}

impl CsvWriter {
    pub fn new(path: PathBuf) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        let mut writer = csv::Writer::from_writer(file);
        writer.write_record([
            "content",
            "source_index",
            "category",
            "input_tokens",
            "output_tokens",
        ])?;
        Ok(Self { writer })
    }
}

impl OutputWriter for CsvWriter {
    fn write(&mut self, result: &GenerationResult) -> Result<()> {
        self.writer.write_record([
            &result.content,
            &result
                .source_index
                .map(|i| i.to_string())
                .unwrap_or_default(),
            &result.category.clone().unwrap_or_default(),
            &result.input_tokens.to_string(),
            &result.output_tokens.to_string(),
        ])?;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

pub struct ParquetWriter {
    results: Vec<GenerationResult>,
    path: PathBuf,
}

impl ParquetWriter {
    pub fn new(path: PathBuf) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(Self {
            results: Vec::new(),
            path,
        })
    }
}

impl OutputWriter for ParquetWriter {
    fn write(&mut self, result: &GenerationResult) -> Result<()> {
        self.results.push(GenerationResult {
            content: result.content.clone(),
            source_index: result.source_index,
            category: result.category.clone(),
            input_tokens: result.input_tokens,
            output_tokens: result.output_tokens,
        });
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.results.is_empty() {
            return Ok(());
        }

        let content: Vec<String> = self.results.iter().map(|r| r.content.clone()).collect();
        let source_index: Vec<Option<i64>> = self
            .results
            .iter()
            .map(|r| r.source_index.map(|i| i as i64))
            .collect();
        let category: Vec<Option<String>> =
            self.results.iter().map(|r| r.category.clone()).collect();
        let input_tokens: Vec<u32> = self.results.iter().map(|r| r.input_tokens).collect();
        let output_tokens: Vec<u32> = self.results.iter().map(|r| r.output_tokens).collect();

        let df = DataFrame::new(vec![
            Series::new("content".into(), content).into(),
            Series::new("source_index".into(), source_index).into(),
            Series::new("category".into(), category).into(),
            Series::new("input_tokens".into(), input_tokens).into(),
            Series::new("output_tokens".into(), output_tokens).into(),
        ])
        .map_err(|e| Error::Dataset(e.to_string()))?;

        let file = File::create(&self.path)?;
        ParquetWriter::write_parquet(df, file)?;

        Ok(())
    }
}

impl ParquetWriter {
    fn write_parquet(df: DataFrame, file: File) -> Result<()> {
        polars::prelude::ParquetWriter::new(file)
            .finish(&mut df.clone())
            .map_err(|e| Error::Dataset(e.to_string()))?;
        Ok(())
    }
}

pub fn create_writer(
    format: &crate::config::OutputFormat,
    path: PathBuf,
) -> Result<Box<dyn OutputWriter>> {
    match format {
        crate::config::OutputFormat::Jsonl => Ok(Box::new(JsonlWriter::new(path)?)),
        crate::config::OutputFormat::Csv => Ok(Box::new(CsvWriter::new(path)?)),
        crate::config::OutputFormat::Parquet => Ok(Box::new(ParquetWriter::new(path)?)),
        crate::config::OutputFormat::Json => Err(Error::Config(
            "JSON array output not yet implemented, use JSONL instead".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn sample_result() -> GenerationResult {
        GenerationResult {
            content: "Test content".to_string(),
            source_index: Some(42),
            category: Some("test".to_string()),
            input_tokens: 10,
            output_tokens: 20,
        }
    }

    #[test]
    fn test_jsonl_writer() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        let mut writer = JsonlWriter::new(path.clone()).unwrap();
        writer.write(&sample_result()).unwrap();
        writer.flush().unwrap();
        drop(writer);

        let content = std::fs::read_to_string(path).unwrap();
        assert!(content.contains("Test content"));
        assert!(content.contains("\"source_index\":42"));
    }

    #[test]
    fn test_csv_writer() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        let mut writer = CsvWriter::new(path.clone()).unwrap();
        writer.write(&sample_result()).unwrap();
        writer.flush().unwrap();
        drop(writer);

        let content = std::fs::read_to_string(path).unwrap();
        assert!(content.contains("Test content"));
        assert!(content.contains("42"));
    }

    #[test]
    fn test_parquet_writer() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        let mut writer = ParquetWriter::new(path.clone()).unwrap();
        writer.write(&sample_result()).unwrap();
        writer.write(&sample_result()).unwrap();
        writer.flush().unwrap();

        let file = File::open(path).unwrap();
        let df = polars::prelude::ParquetReader::new(file).finish().unwrap();
        assert_eq!(df.height(), 2);
        assert!(df
            .get_column_names()
            .iter()
            .any(|s| s.as_str() == "content"));
    }
}
