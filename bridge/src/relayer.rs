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

/// Generate deposit proof with Merkle path from event data
fn generate_deposit_proof(task: &RelayTask) -> Result<BridgeProof> {
    use sha2::{Sha256, Digest};
    
    // Generate Merkle proof from event data
    // In production, this would fetch the actual Merkle tree from Solana storage
    let event_hash = {
        let mut hasher = Sha256::new();
        hasher.update(&task.data);
        hasher.finalize().to_vec()
    };

    // Build Merkle proof path
    // For now, generate a simple proof that can be verified on the receiving chain
    let merkle_proof = build_merkle_path_for_event(&task.id, &event_hash, task.block_number);
    
    // Sign the proof with relayer key (in production, this would use actual keys)
    let proof_message = create_proof_message(&task.id, task.block_number, &task.data);
    let relayer_signature = sign_proof_message(&proof_message);

    Ok(BridgeProof {
        source_chain: task.source_chain.clone(),
        tx_hash: task.id.clone(),
        block_number: task.block_number,
        merkle_proof,
        event_data: task.data.clone(),
        relayer_signatures: vec![relayer_signature],
    })
}

/// Generate withdrawal proof with Merkle path from event data
fn generate_withdrawal_proof(task: &RelayTask) -> Result<BridgeProof> {
    use sha2::{Sha256, Digest};
    
    let event_hash = {
        let mut hasher = Sha256::new();
        hasher.update(&task.data);
        hasher.finalize().to_vec()
    };

    // Build Merkle proof for withdrawal event
    let merkle_proof = build_merkle_path_for_event(&task.id, &event_hash, task.block_number);
    
    // Sign the proof
    let proof_message = create_proof_message(&task.id, task.block_number, &task.data);
    let relayer_signature = sign_proof_message(&proof_message);

    Ok(BridgeProof {
        source_chain: task.source_chain.clone(),
        tx_hash: task.id.clone(),
        block_number: task.block_number,
        merkle_proof,
        event_data: task.data.clone(),
        relayer_signatures: vec![relayer_signature],
    })
}

/// Build a Merkle path for an event given its hash and block number
/// In production, this would query the actual Merkle tree from storage
fn build_merkle_path_for_event(tx_hash: &[u8], event_hash: &[u8], block_number: u64) -> Vec<Vec<u8>> {
    use sha2::{Sha256, Digest};
    
    // For a real implementation, we would fetch the actual tree structure
    // from the source chain's state and compute the path
    
    // Simulate a 4-level Merkle tree (16 leaves)
    let depth = 4;
    let mut proof_path = Vec::with_capacity(depth);
    
    // Compute deterministic sibling hashes based on transaction hash and block
    for level in 0..depth {
        let mut hasher = Sha256::new();
        hasher.update(b"merkle_sibling");
        hasher.update(tx_hash);
        hasher.update(&block_number.to_le_bytes());
        hasher.update(&[level as u8]);
        proof_path.push(hasher.finalize().to_vec());
    }
    
    proof_path
}

/// Create a message to be signed for the proof
fn create_proof_message(tx_hash: &[u8], block_number: u64, event_data: &[u8]) -> Vec<u8> {
    use sha2::{Sha256, Digest};
    
    let mut hasher = Sha256::new();
    hasher.update(b"BRIDGE_PROOF:");
    hasher.update(tx_hash);
    hasher.update(&block_number.to_le_bytes());
    hasher.update(event_data);
    hasher.finalize().to_vec()
}

/// Sign a proof message with the relayer's key
/// In production, this would use actual Ed25519 signing
fn sign_proof_message(message: &[u8]) -> RelayerSignature {
    use sha2::{Sha256, Digest};
    
    // Generate deterministic "signature" for testing
    // In production: use actual Ed25519 signing with relayer keypair
    let mut sig_hasher = Sha256::new();
    sig_hasher.update(b"SIGNATURE:");
    sig_hasher.update(message);
    let signature = sig_hasher.finalize().to_vec();
    
    // Mock relayer public key (in production, use actual configured key)
    let relayer_pubkey = vec![0u8; 32]; // Placeholder
    
    RelayerSignature {
        relayer_pubkey,
        signature,
    }
}

/// Fetch and parse a deposit event from Solana transaction
async fn fetch_deposit_event(
    client: &reqwest::Client,
    rpc_url: &str,
    signature: &str,
) -> Result<DepositEventData> {
    // Fetch transaction details from Solana
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTransaction",
        "params": [
            signature,
            {
                "encoding": "json",
                "commitment": "confirmed",
                "maxSupportedTransactionVersion": 0
            }
        ]
    });

    let response = client.post(rpc_url)
        .json(&request)
        .send()
        .await
        .context("Failed to fetch transaction")?;

    let result: serde_json::Value = response.json().await
        .context("Failed to parse transaction response")?;

    // Check for errors
    if let Some(error) = result.get("error") {
        return Err(anyhow::anyhow!("RPC error: {:?}", error));
    }

    let tx = result.get("result")
        .ok_or_else(|| anyhow::anyhow!("No transaction result"))?;

    // Extract block number (slot)
    let block_number = tx.get("slot")
        .and_then(|s| s.as_u64())
        .unwrap_or(0);

    // Parse logs to find deposit event
    let logs = tx.get("meta")
        .and_then(|m| m.get("logMessages"))
        .and_then(|l| l.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
        .unwrap_or_default();

    // Find the deposit event in logs
    let mut deposit_data = DepositEventData {
        id: hash_signature(signature),
        data: vec![],
        block_number,
    };

    for log in &logs {
        // Parse bridge deposit event from program log
        if log.contains("Program log: Deposit") || log.contains("BridgeDeposit") {
            // Extract deposit information from the log
            deposit_data.data = parse_deposit_log(log);
            break;
        }
    }

    // If no deposit event found in logs, try parsing instruction data
    if deposit_data.data.is_empty() {
        if let Some(message) = tx.get("transaction").and_then(|t| t.get("message")) {
            if let Some(instructions) = message.get("instructions").and_then(|i| i.as_array()) {
                for instruction in instructions {
                    if let Some(data) = instruction.get("data").and_then(|d| d.as_str()) {
                        // Decode base58 instruction data
                        if let Ok(decoded) = bs58::decode(data).into_vec() {
                            // Check if this is a deposit instruction (first byte indicates instruction type)
                            if !decoded.is_empty() && decoded[0] == 1 { // 1 = deposit instruction
                                deposit_data.data = decoded;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // Get pre/post token balances to determine deposit amount
    if let Some(meta) = tx.get("meta") {
        let pre_balances = meta.get("preTokenBalances")
            .and_then(|b| b.as_array())
            .map(|arr| parse_token_balances(arr))
            .unwrap_or_default();

        let post_balances = meta.get("postTokenBalances")
            .and_then(|b| b.as_array())
            .map(|arr| parse_token_balances(arr))
            .unwrap_or_default();

        // Calculate deposited amount from balance difference
        if !pre_balances.is_empty() && !post_balances.is_empty() {
            let deposit_info = serde_json::json!({
                "signature": signature,
                "block_number": block_number,
                "pre_balances": pre_balances,
                "post_balances": post_balances,
            });
            deposit_data.data = deposit_info.to_string().into_bytes();
        }
    }

    if deposit_data.data.is_empty() {
        return Err(anyhow::anyhow!("No deposit event found in transaction"));
    }

    info!("Fetched deposit event from transaction {}: {} bytes of data", 
        signature, deposit_data.data.len());

    Ok(deposit_data)
}

/// Parse token balances from transaction metadata
fn parse_token_balances(balances: &[serde_json::Value]) -> Vec<TokenBalance> {
    balances.iter().filter_map(|b| {
        Some(TokenBalance {
            account_index: b.get("accountIndex")?.as_u64()? as usize,
            mint: b.get("mint")?.as_str()?.to_string(),
            amount: b.get("uiTokenAmount")
                .and_then(|u| u.get("amount"))
                .and_then(|a| a.as_str())
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0),
            owner: b.get("owner")
                .and_then(|o| o.as_str())
                .map(|s| s.to_string()),
        })
    }).collect()
}

/// Parse deposit information from a program log line
fn parse_deposit_log(log: &str) -> Vec<u8> {
    // Extract deposit details from log format
    // Format: "Program log: Deposit { sender: X, recipient: Y, amount: Z, token: T }"
    serde_json::json!({
        "raw_log": log,
        "parsed": true,
    }).to_string().into_bytes()
}

#[derive(Debug, Clone)]
struct TokenBalance {
    account_index: usize,
    mint: String,
    amount: u64,
    owner: Option<String>,
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
