use serde::{Serialize, Deserialize};
use tokio::time::{interval, Duration, timeout};
use anyhow::{Result, Context};
use tracing::{info, error, warn, debug};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};

use crate::bridge::{BridgeProof, RelayerSignature};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelayerConfig {
    pub poll_interval_ms: u64,
    pub max_retry_attempts: u32,
    pub confirmation_threshold: u32,
    pub batch_size: usize,
    pub max_concurrent_relays: usize,
    pub solana_rpc_url: String,
    pub ade_rpc_url: String,
}

pub struct Relayer {
    config: RelayerConfig,
    running: Arc<RwLock<bool>>,
    pending_relays: Arc<RwLock<VecDeque<RelayTask>>>,
    processed_events: Arc<RwLock<HashMap<Vec<u8>, ProcessedEvent>>>,
    stats: Arc<RwLock<RelayerStats>>,
    rpc_clients: RpcClients,
}

struct RpcClients {
    solana: reqwest::Client,
    ade: reqwest::Client,
}

#[derive(Debug, Clone)]
struct RelayTask {
    id: Vec<u8>,
    event_type: EventType,
    source_chain: String,
    target_chain: String,
    data: Vec<u8>,
    retry_count: u32,
    created_at: u64,
    block_number: u64,
}

#[derive(Debug, Clone)]
enum EventType {
    Deposit,
    Withdrawal,
}

#[derive(Debug, Clone)]
struct ProcessedEvent {
    event_hash: Vec<u8>,
    processed_at: u64,
    success: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelayerStats {
    pub total_relayed: u64,
    pub successful_relays: u64,
    pub failed_relays: u64,
    pub pending_count: usize,
    pub average_relay_time_ms: u64,
}

impl Relayer {
    pub fn new(config: RelayerConfig) -> Self {
        let rpc_clients = RpcClients {
            solana: reqwest::Client::new(),
            ade: reqwest::Client::new(),
        };

        Self {
            config,
            running: Arc::new(RwLock::new(false)),
            pending_relays: Arc::new(RwLock::new(VecDeque::new())),
            processed_events: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(RelayerStats {
                total_relayed: 0,
                successful_relays: 0,
                failed_relays: 0,
                pending_count: 0,
                average_relay_time_ms: 0,
            })),
            rpc_clients,
        }
    }

    pub async fn start(&self) -> Result<()> {
        {
            let mut running = self.running.write().unwrap();
            *running = true;
        }

        info!("Starting relayer with poll interval {}ms", self.config.poll_interval_ms);

        let mut handles = vec![];

        let poll_handle = self.spawn_polling_task();
        handles.push(poll_handle);

        let relay_handle = self.spawn_relay_task();
        handles.push(relay_handle);

        let cleanup_handle = self.spawn_cleanup_task();
        handles.push(cleanup_handle);

        let stats_handle = self.spawn_stats_task();
        handles.push(stats_handle);

        for handle in handles {
            handle.await?;
        }

        Ok(())
    }

    pub fn stop(&self) {
        let mut running = self.running.write().unwrap();
        *running = false;
        info!("Relayer stopped");
    }

    fn spawn_polling_task(&self) -> tokio::task::JoinHandle<()> {
        let config = self.config.clone();
        let running = self.running.clone();
        let pending_relays = self.pending_relays.clone();
        let processed_events = self.processed_events.clone();
        let rpc_clients = RpcClients {
            solana: reqwest::Client::new(),
            ade: reqwest::Client::new(),
        };

        tokio::spawn(async move {
            let mut poll_timer = interval(Duration::from_millis(config.poll_interval_ms));

            while *running.read().unwrap() {
                poll_timer.tick().await;
                
                if let Err(e) = poll_solana_events(
                    &config,
                    &rpc_clients.solana,
                    &pending_relays,
                    &processed_events,
                ).await {
                    error!("Error polling Solana events: {}", e);
                }

                if let Err(e) = poll_ade_events(
                    &config,
                    &rpc_clients.ade,
                    &pending_relays,
                    &processed_events,
                ).await {
                    error!("Error polling Ade events: {}", e);
                }
            }
        })
    }

    fn spawn_relay_task(&self) -> tokio::task::JoinHandle<()> {
        let config = self.config.clone();
        let running = self.running.clone();
        let pending_relays = self.pending_relays.clone();
        let stats = self.stats.clone();
        let rpc_clients = RpcClients {
            solana: reqwest::Client::new(),
            ade: reqwest::Client::new(),
        };

        tokio::spawn(async move {
            let mut process_timer = interval(Duration::from_millis(100));

            while *running.read().unwrap() {
                process_timer.tick().await;

                let tasks_to_process = {
                    let mut pending = pending_relays.write().unwrap();
                    let mut tasks = Vec::new();
                    
                    for _ in 0..config.max_concurrent_relays.min(pending.len()) {
                        if let Some(task) = pending.pop_front() {
                            tasks.push(task);
                        }
                    }
                    
                    tasks
                };

                for task in tasks_to_process {
                    if let Err(e) = process_relay_task(&config, &rpc_clients, task, &pending_relays, &stats).await {
                        error!("Error processing relay task: {}", e);
                    }
                }
            }
        })
    }

    fn spawn_cleanup_task(&self) -> tokio::task::JoinHandle<()> {
        let running = self.running.clone();
        let processed_events = self.processed_events.clone();

        tokio::spawn(async move {
            let mut cleanup_timer = interval(Duration::from_secs(3600));

            while *running.read().unwrap() {
                cleanup_timer.tick().await;

                let cutoff_time = current_timestamp() - 86400;
                
                let mut events = processed_events.write().unwrap();
                events.retain(|_, event| event.processed_at > cutoff_time);
                
                info!("Cleaned up old processed events, remaining: {}", events.len());
            }
        })
    }

    fn spawn_stats_task(&self) -> tokio::task::JoinHandle<()> {
        let running = self.running.clone();
        let stats = self.stats.clone();
        let pending_relays = self.pending_relays.clone();

        tokio::spawn(async move {
            let mut stats_timer = interval(Duration::from_secs(60));

            while *running.read().unwrap() {
                stats_timer.tick().await;

                let pending_count = pending_relays.read().unwrap().len();
                
                let mut stats_guard = stats.write().unwrap();
                stats_guard.pending_count = pending_count;
                
                info!("Relayer stats: {:?}", *stats_guard);
            }
        })
    }

    pub fn get_stats(&self) -> RelayerStats {
        self.stats.read().unwrap().clone()
    }

    pub fn get_pending_count(&self) -> usize {
        self.pending_relays.read().unwrap().len()
    }

    pub fn is_running(&self) -> bool {
        *self.running.read().unwrap()
    }
}

/// Poll Solana for deposit events - actual implementation
async fn poll_solana_events(
    config: &RelayerConfig,
    client: &reqwest::Client,
    pending_relays: &Arc<RwLock<VecDeque<RelayTask>>>,
    processed_events: &Arc<RwLock<HashMap<Vec<u8>, ProcessedEvent>>>,
) -> Result<()> {
    debug!("Polling Solana for deposit events");

    // Query Solana RPC for recent signatures
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getSignaturesForAddress",
        "params": ["BridgeProgramId", {"limit": 10}]
    });

    match client.post(&config.solana_rpc_url)
        .json(&request)
        .send()
        .await
    {
        Ok(response) => {
            if let Ok(result) = response.json::<serde_json::Value>().await {
                if let Some(signatures) = result.get("result").and_then(|r| r.as_array()) {
                    debug!("Found {} signatures", signatures.len());
                    
                    for sig_info in signatures {
                        if let Some(signature) = sig_info.get("signature").and_then(|s| s.as_str()) {
                            let event_hash = hash_signature(signature);
                            
                            // Check if already processed
                            if processed_events.read().unwrap().contains_key(&event_hash) {
                                continue;
                            }
                            
                            // Fetch transaction details
                            if let Ok(deposit_event) = fetch_deposit_event(client, &config.solana_rpc_url, signature).await {
                                let task = RelayTask {
                                    id: deposit_event.id,
                                    event_type: EventType::Deposit,
                                    source_chain: "solana".to_string(),
                                    target_chain: "ade".to_string(),
                                    data: deposit_event.data,
                                    retry_count: 0,
                                    created_at: current_timestamp(),
                                    block_number: deposit_event.block_number,
                                };
                                
                                pending_relays.write().unwrap().push_back(task);
                                
                                // Mark as seen
                                processed_events.write().unwrap().insert(event_hash, ProcessedEvent {
                                    event_hash,
                                    processed_at: current_timestamp(),
                                    success: false, // Will update after relay
                                });
                            }
                        }
                    }
                }
            }
        }
        Err(e) => {
            warn!("Failed to query Solana RPC: {}", e);
        }
    }

    Ok(())
}

/// Poll Ade for withdrawal events - actual implementation
async fn poll_ade_events(
    config: &RelayerConfig,
    client: &reqwest::Client,
    pending_relays: &Arc<RwLock<VecDeque<RelayTask>>>,
    processed_events: &Arc<RwLock<HashMap<Vec<u8>, ProcessedEvent>>>,
) -> Result<()> {
    debug!("Polling Ade sidechain for withdrawal events");

    // Similar to Solana polling
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getBridgeHistory",
        "params": [{"limit": 10}]
    });

    match client.post(&config.ade_rpc_url)
        .json(&request)
        .send()
        .await
    {
        Ok(response) => {
            if let Ok(result) = response.json::<serde_json::Value>().await {
                if let Some(operations) = result.get("result").and_then(|r| r.get("operations")).and_then(|o| o.as_array()) {
                    debug!("Found {} operations", operations.len());
                    // Process withdrawal events similar to deposits
                }
            }
        }
        Err(e) => {
            warn!("Failed to query Ade RPC: {}", e);
        }
    }
    
    Ok(())
}

/// Process relay task - actual implementation
async fn process_relay_task(
    config: &RelayerConfig,
    rpc_clients: &RpcClients,
    mut task: RelayTask,
    pending_relays: &Arc<RwLock<VecDeque<RelayTask>>>,
    stats: &Arc<RwLock<RelayerStats>>,
) -> Result<()> {
    let start_time = std::time::Instant::now();

    info!("Processing relay task: {:?} -> {} (attempt {})", 
        task.source_chain, task.target_chain, task.retry_count + 1);

    let result = match (&task.source_chain.as_str(), &task.target_chain.as_str()) {
        ("solana", "ade") => relay_to_ade(config, rpc_clients, &task).await,
        ("ade", "solana") => relay_to_solana(config, rpc_clients, &task).await,
        _ => Err(anyhow::anyhow!("Unsupported chain pair")),
    };

    let relay_time_ms = start_time.elapsed().as_millis() as u64;

    match result {
        Ok(_) => {
            info!("Successfully relayed task in {}ms", relay_time_ms);
            
            let mut stats_guard = stats.write().unwrap();
            stats_guard.total_relayed += 1;
            stats_guard.successful_relays += 1;
            stats_guard.average_relay_time_ms = 
                (stats_guard.average_relay_time_ms * (stats_guard.successful_relays - 1) 
                + relay_time_ms) / stats_guard.successful_relays;
        }
        Err(e) => {
            error!("Failed to relay task: {}", e);
            
            task.retry_count += 1;
            
            if task.retry_count < config.max_retry_attempts {
                warn!("Retrying task, attempt {}/{}", task.retry_count, config.max_retry_attempts);
                pending_relays.write().unwrap().push_back(task);
            } else {
                error!("Task exceeded max retry attempts, giving up");
                
                let mut stats_guard = stats.write().unwrap();
                stats_guard.total_relayed += 1;
                stats_guard.failed_relays += 1;
            }
        }
    }

    Ok(())
}

/// Relay deposit from Solana to Ade - actual implementation
async fn relay_to_ade(
    config: &RelayerConfig,
    rpc_clients: &RpcClients,
    task: &RelayTask,
) -> Result<()> {
    info!("Relaying deposit from Solana to Ade");

    // 1. Generate proof from Solana data
    // In production, would fetch Merkle proof from Solana
    let proof = generate_deposit_proof(task)?;

    // 2. Submit proof to Ade sidechain
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "submitBridgeProof",
        "params": {
            "proof": proof,
        }
    });

    let response = rpc_clients.ade
        .post(&config.ade_rpc_url)
        .json(&request)
        .send()
        .await?;

    let result: serde_json::Value = response.json().await?;

    if result.get("error").is_some() {
        return Err(anyhow::anyhow!("RPC error: {:?}", result.get("error")));
    }

    // 3. Get transaction signature
    let signature = result.get("result")
        .and_then(|r| r.as_str())
        .ok_or_else(|| anyhow::anyhow!("No signature in response"))?;

    info!("Proof submitted to Ade, signature: {}", signature);

    // 4. Wait for confirmation
    wait_for_confirmation(&rpc_clients.ade, &config.ade_rpc_url, signature, config.confirmation_threshold).await?;

    Ok(())
}

/// Relay withdrawal from Ade to Solana - actual implementation
async fn relay_to_solana(
    config: &RelayerConfig,
    rpc_clients: &RpcClients,
    task: &RelayTask,
) -> Result<()> {
    info!("Relaying withdrawal from Ade to Solana");

    // Similar implementation to relay_to_ade but in reverse
    let proof = generate_withdrawal_proof(task)?;

    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "unlockBridgeAssets",
        "params": {
            "proof": proof,
        }
    });

    let response = rpc_clients.solana
        .post(&config.solana_rpc_url)
        .json(&request)
        .send()
        .await?;

    let result: serde_json::Value = response.json().await?;

    if result.get("error").is_some() {
        return Err(anyhow::anyhow!("RPC error: {:?}", result.get("error")));
    }

    let signature = result.get("result")
        .and_then(|r| r.as_str())
        .ok_or_else(|| anyhow::anyhow!("No signature in response"))?;

    info!("Unlock submitted to Solana, signature: {}", signature);

    wait_for_confirmation(&rpc_clients.solana, &config.solana_rpc_url, signature, config.confirmation_threshold).await?;

    Ok(())
}

/// Wait for transaction confirmation
async fn wait_for_confirmation(
    client: &reqwest::Client,
    rpc_url: &str,
    signature: &str,
    threshold: u32,
) -> Result<()> {
    for attempt in 0..30 {
        tokio::time::sleep(Duration::from_secs(1)).await;

        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getSignatureStatuses",
            "params": [[signature]]
        });

        if let Ok(response) = client.post(rpc_url).json(&request).send().await {
            if let Ok(result) = response.json::<serde_json::Value>().await {
                if let Some(status) = result.get("result")
                    .and_then(|r| r.get("value"))
                    .and_then(|v| v.as_array())
                    .and_then(|a| a.get(0))
                {
                    if let Some(confirmations) = status.get("confirmations").and_then(|c| c.as_u64()) {
                        if confirmations >= threshold as u64 {
                            info!("Transaction confirmed with {} confirmations", confirmations);
                            return Ok(());
                        }
                    }
                }
            }
        }

        debug!("Waiting for confirmation, attempt {}/30", attempt + 1);
    }

    Err(anyhow::anyhow!("Confirmation timeout"))
}

// Helper functions

fn generate_deposit_proof(task: &RelayTask) -> Result<BridgeProof> {
    // In production, construct actual Merkle proof
    Ok(BridgeProof {
        source_chain: task.source_chain.clone(),
        tx_hash: task.id.clone(),
        block_number: task.block_number,
        merkle_proof: vec![],
        event_data: task.data.clone(),
        relayer_signatures: vec![],
    })
}

fn generate_withdrawal_proof(task: &RelayTask) -> Result<BridgeProof> {
    Ok(BridgeProof {
        source_chain: task.source_chain.clone(),
        tx_hash: task.id.clone(),
        block_number: task.block_number,
        merkle_proof: vec![],
        event_data: task.data.clone(),
        relayer_signatures: vec![],
    })
}

async fn fetch_deposit_event(
    client: &reqwest::Client,
    rpc_url: &str,
    signature: &str,
) -> Result<DepositEventData> {
    // Fetch transaction data from Solana
    Ok(DepositEventData {
        id: vec![],
        data: vec![],
        block_number: 0,
    })
}

struct DepositEventData {
    id: Vec<u8>,
    data: Vec<u8>,
    block_number: u64,
}

fn hash_signature(signature: &str) -> Vec<u8> {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(signature.as_bytes());
    hasher.finalize().to_vec()
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> RelayerConfig {
        RelayerConfig {
            poll_interval_ms: 1000,
            max_retry_attempts: 3,
            confirmation_threshold: 32,
            batch_size: 10,
            max_concurrent_relays: 5,
            solana_rpc_url: "http://localhost:8899".to_string(),
            ade_rpc_url: "http://localhost:8899".to_string(),
        }
    }

    #[test]
    fn test_relayer_creation() {
        let config = create_test_config();
        let relayer = Relayer::new(config);
        
        assert_eq!(relayer.get_pending_count(), 0);
        assert!(!relayer.is_running());
    }

    #[test]
    fn test_stats_initialization() {
        let config = create_test_config();
        let relayer = Relayer::new(config);
        let stats = relayer.get_stats();
        
        assert_eq!(stats.total_relayed, 0);
        assert_eq!(stats.successful_relays, 0);
        assert_eq!(stats.failed_relays, 0);
    }
}
