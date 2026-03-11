use crate::config::HubConfig;
use crate::{Error, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;

const HF_API_URL: &str = "https://huggingface.co/api";

pub struct HubClient {
    client: Client,
    token: String,
}

#[derive(Serialize)]
struct CreateRepoRequest {
    #[serde(rename = "type")]
    repo_type: String,
    name: String,
    private: bool,
}

#[derive(Deserialize)]
struct RepoInfo {
    pub name: String,
}

fn resolve_token(token: Option<String>) -> Result<String> {
    token
        .or_else(|| std::env::var("HF_TOKEN").ok())
        .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok())
        .or_else(read_cached_token)
        .ok_or_else(|| Error::Config(
            "HF token not found. Set via config, HF_TOKEN env var, or run `huggingface-cli login`".into()
        ))
}

fn read_cached_token() -> Option<String> {
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).ok()?;
    let token_path = std::path::PathBuf::from(home).join(".cache/huggingface/token");
    std::fs::read_to_string(token_path).ok().map(|s| s.trim().to_string())
}

impl HubClient {
    pub fn new(token: Option<String>) -> Result<Self> {
        Ok(Self {
            client: Client::new(),
            token: resolve_token(token)?,
        })
    }

    pub fn from_config(config: &HubConfig) -> Result<Self> {
        Self::new(config.token.clone())
    }

    pub async fn create_dataset_repo(&self, repo_id: &str, private: bool) -> Result<String> {
        let repo_name = repo_id.split('/').last().unwrap_or(repo_id);

        let req = CreateRepoRequest {
            repo_type: "dataset".to_string(),
            name: repo_name.to_string(),
            private,
        };

        let resp = self
            .client
            .post(&format!("{}/repos/create", HF_API_URL))
            .header("Authorization", format!("Bearer {}", self.token))
            .json(&req)
            .send()
            .await
            .map_err(|e| Error::Http(e))?;

        if resp.status() == 409 {
            if repo_id.contains('/') {
                return Ok(repo_id.to_string());
            }
            return Err(Error::Config(format!(
                "Dataset '{}' already exists. Use 'username/{}' in hub.repo",
                repo_name, repo_name
            )));
        }

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(Error::Provider(format!("Failed to create repo: {} - {}", status, body)));
        }

        let info: RepoInfo = resp.json().await.map_err(|e| Error::Provider(e.to_string()))?;
        // HF needs time to propagate newly created repos
        tokio::time::sleep(Duration::from_secs(2)).await;
        Ok(info.name)
    }

    pub async fn upload_file(
        &self,
        repo_id: &str,
        path_in_repo: &str,
        content: &[u8],
        commit_message: Option<&str>,
    ) -> Result<String> {
        let commit_msg = commit_message.unwrap_or("Upload file");

        use base64::Engine;
        let content_b64 = base64::engine::general_purpose::STANDARD.encode(content);

        // HF commit API uses NDJSON (application/x-ndjson)
        let header_line = serde_json::json!({
            "key": "header",
            "value": { "summary": commit_msg }
        });
        let file_line = serde_json::json!({
            "key": "file",
            "value": {
                "path": path_in_repo,
                "content": content_b64,
                "encoding": "base64"
            }
        });
        let body = format!("{}\n{}", header_line, file_line);

        let url = format!("{}/datasets/{}/commit/main", HF_API_URL, repo_id);

        let mut resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Content-Type", "application/x-ndjson")
            .body(body.clone())
            .send()
            .await
            .map_err(Error::Http)?;

        // Retry on 404 — newly created repos need time to propagate
        let mut attempts = 0;
        while resp.status() == 404 && attempts < 5 {
            attempts += 1;
            tokio::time::sleep(Duration::from_secs(1)).await;
            resp = self
                .client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.token))
                .header("Content-Type", "application/x-ndjson")
                .body(body.clone())
                .send()
                .await
                .map_err(Error::Http)?;
        }

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(Error::Provider(format!("Upload failed: {} - {}", status, body)));
        }

        Ok(format!(
            "https://huggingface.co/datasets/{}/blob/main/{}",
            repo_id, path_in_repo
        ))
    }

    pub async fn upload_file_from_path(
        &self,
        repo_id: &str,
        local_path: &Path,
        path_in_repo: &str,
        commit_message: Option<&str>,
    ) -> Result<String> {
        let content = std::fs::read(local_path)
            .map_err(|e| Error::Io(e))?;
        self.upload_file(repo_id, path_in_repo, &content, commit_message).await
    }
}

pub struct DatasetUploader {
    client: HubClient,
    repo_id: String,
}

impl DatasetUploader {
    pub async fn new(repo_name: &str, private: bool, token: Option<String>) -> Result<Self> {
        let client = HubClient::new(token)?;
        let repo_id = client.create_dataset_repo(repo_name, private).await?;
        Ok(Self { client, repo_id })
    }

    pub async fn from_config(config: &HubConfig) -> Result<Self> {
        let repo = config.repo.as_ref()
            .ok_or_else(|| Error::Config("hub.repo is required".into()))?;
        Self::new(repo, config.private, config.token.clone()).await
    }

    pub fn repo_id(&self) -> &str {
        &self.repo_id
    }

    pub fn repo_url(&self) -> String {
        format!("https://huggingface.co/datasets/{}", self.repo_id)
    }

    pub async fn upload(&self, path_in_repo: &str, content: &[u8], commit_message: Option<&str>) -> Result<String> {
        self.client.upload_file(&self.repo_id, path_in_repo, content, commit_message).await
    }

    pub async fn upload_file(&self, local_path: &Path, path_in_repo: &str, commit_message: Option<&str>) -> Result<String> {
        self.client.upload_file_from_path(&self.repo_id, local_path, path_in_repo, commit_message).await
    }

    pub async fn upload_jsonl(&self, data: &[serde_json::Value], filename: &str) -> Result<String> {
        let content: String = data
            .iter()
            .map(|v| serde_json::to_string(v).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        
        self.upload(filename, content.as_bytes(), Some(&format!("Upload {}", filename))).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hub_client_with_token() {
        let result = HubClient::new(Some("test-token".to_string()));
        assert!(result.is_ok());
    }

    #[test]
    fn test_resolve_token_priority() {
        std::env::set_var("HF_TOKEN", "env-token");
        let token = resolve_token(Some("direct-token".to_string())).unwrap();
        assert_eq!(token, "direct-token");
        std::env::remove_var("HF_TOKEN");
    }
}
