use anyhow::Result;
use serde_json::{json, Value};

pub struct RpcClient {
    client: reqwest::Client,
    url: String,
    request_id: std::sync::atomic::AtomicU64,
}

impl RpcClient {
    pub fn new(url: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            url: url.to_string(),
            request_id: std::sync::atomic::AtomicU64::new(1),
        }
    }

    pub async fn call(&self, method: &str, params: Option<Value>) -> Result<Value> {
        let id = self.request_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        let request = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let response: Value = self.client
            .post(&self.url)
            .json(&request)
            .send()
            .await?
            .json()
            .await?;

        if let Some(error) = response.get("error") {
            return Err(anyhow::anyhow!("RPC Error: {}", error));
        }

        response.get("result")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("No result in response"))
    }
}

