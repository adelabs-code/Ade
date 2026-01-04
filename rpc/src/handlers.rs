use axum::Json;
use crate::server::RpcState;
use crate::types::RpcResponse;
use crate::state::{TransactionInfo, TransactionStatus, AIAgentData};
use serde_json::json;
use sha2::{Sha256, Digest};
use tracing::{info, debug, warn};

// ============================================================================
// Block Methods
// ============================================================================

pub async fn get_slot(state: RpcState) -> Json<RpcResponse> {
    let chain = state.chain_state.read().await;
    Json(RpcResponse::success(json!(chain.current_slot)))
}

pub async fn get_block_height(state: RpcState) -> Json<RpcResponse> {
    let chain = state.chain_state.read().await;
    Json(RpcResponse::success(json!(chain.current_slot)))
}

pub async fn get_block(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    let slot = params
        .and_then(|p| p.get("slot"))
        .and_then(|s| s.as_u64())
        .unwrap_or(0);

    // Try to get actual block from storage
    let blocks = state.blocks.read().await;
    
    if let Some(block) = blocks.get(&slot) {
        // Return actual block data
        let transactions: Vec<_> = block.transactions.iter().map(|tx| {
            json!({
                "transaction": {
                    "signatures": [bs58::encode(tx.signature().unwrap_or(&[])).into_string()],
                    "message": {
                        "header": {
                            "numRequiredSignatures": 1,
                            "numReadonlySignedAccounts": 0,
                            "numReadonlyUnsignedAccounts": 0
                        },
                        "accountKeys": tx.message.account_keys.iter()
                            .map(|k| bs58::encode(k).into_string())
                            .collect::<Vec<_>>(),
                        "recentBlockhash": bs58::encode(&tx.message.recent_blockhash).into_string(),
                        "instructions": tx.message.instructions.iter()
                            .map(|ix| json!({
                                "programIdIndex": ix.program_id_index,
                                "accounts": ix.accounts.iter().map(|a| a.index).collect::<Vec<_>>(),
                                "data": bs58::encode(&ix.data).into_string()
                            }))
                            .collect::<Vec<_>>()
                    }
                },
                "meta": null
            })
        }).collect();

        let prev_blockhash = if slot > 0 {
            blocks.get(&(slot - 1))
                .map(|b| bs58::encode(&b.hash()).into_string())
                .unwrap_or_else(|| bs58::encode(&[0u8; 32]).into_string())
        } else {
            bs58::encode(&[0u8; 32]).into_string()
        };

        Json(RpcResponse::success(json!({
            "blockhash": bs58::encode(&block.hash()).into_string(),
            "previousBlockhash": prev_blockhash,
            "parentSlot": slot.saturating_sub(1),
            "transactions": transactions,
            "rewards": [],
            "blockTime": block.header.timestamp,
            "blockHeight": slot
        })))
    } else {
        // Block not found, return error or empty response
        let chain = state.chain_state.read().await;
        
        if slot > chain.current_slot {
            Json(RpcResponse::error(&format!("Block {} not available, current slot is {}", slot, chain.current_slot)))
        } else {
            // Block may have been pruned
            Json(RpcResponse::success(json!({
                "blockhash": bs58::encode(&chain.latest_blockhash).into_string(),
                "previousBlockhash": bs58::encode(&[0u8; 32]).into_string(),
                "parentSlot": slot.saturating_sub(1),
                "transactions": [],
                "rewards": [],
                "blockTime": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                "blockHeight": slot
            })))
        }
    }
}

pub async fn get_blocks(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    let start_slot = params
        .as_ref()
        .and_then(|p| p.get("startSlot"))
        .and_then(|s| s.as_u64())
        .unwrap_or(0);
    
    let end_slot = params
        .and_then(|p| p.get("endSlot"))
        .and_then(|s| s.as_u64());

    let chain = state.chain_state.read().await;
    let end = end_slot.unwrap_or(chain.current_slot);
    
    let blocks: Vec<u64> = (start_slot..=end.min(start_slot + 500)).collect();
    
    Json(RpcResponse::success(json!(blocks)))
}

pub async fn get_block_time(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    let slot = match params
        .and_then(|p| p.get("slot"))
        .and_then(|s| s.as_u64())
    {
        Some(s) => s,
        None => return Json(RpcResponse::error("Missing slot parameter")),
    };

    // Try to get block timestamp from storage
    let blocks = state.blocks.read().await;
    
    if let Some(block) = blocks.get(&slot) {
        Json(RpcResponse::success(json!(block.header.timestamp)))
    } else {
        // Block not found, estimate based on slot time (400ms per slot)
        let chain = state.chain_state.read().await;
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // Estimate: current_time - (current_slot - slot) * 0.4 seconds
        let slot_diff = chain.current_slot.saturating_sub(slot);
        let estimated_time = current_time.saturating_sub(slot_diff * 400 / 1000);
        
        Json(RpcResponse::success(json!(estimated_time)))
    }
}

pub async fn get_first_available_block(state: RpcState) -> Json<RpcResponse> {
    Json(RpcResponse::success(json!(0)))
}

pub async fn get_latest_blockhash(state: RpcState) -> Json<RpcResponse> {
    let chain = state.chain_state.read().await;
    let blockhash = bs58::encode(&chain.latest_blockhash).into_string();
    Json(RpcResponse::success(json!({
        "blockhash": blockhash,
        "lastValidBlockHeight": chain.current_slot + 150
    })))
}

pub async fn get_block_production(state: RpcState) -> Json<RpcResponse> {
    Json(RpcResponse::success(json!({
        "byIdentity": {
            "validator1": [200, 5],
            "validator2": [150, 3]
        },
        "range": {
            "firstSlot": 0,
            "lastSlot": 1000
        }
    })))
}

pub async fn get_block_commitment(
    _state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    if params.is_none() {
        return Json(RpcResponse::error("Missing slot parameter"));
    }

    Json(RpcResponse::success(json!({
        "commitment": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32],
        "totalStake": 1000000000
    })))
}

// ============================================================================
// Transaction Methods
// ============================================================================

pub async fn send_transaction(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    let params = match params {
        Some(p) => p,
        None => return Json(RpcResponse::error("Missing transaction parameter")),
    };

    // Extract transaction data
    let tx_data = params.get("transaction")
        .and_then(|t| t.as_str())
        .map(|s| bs58::decode(s).into_vec().unwrap_or_default());

    let tx_bytes = match tx_data {
        Some(bytes) if !bytes.is_empty() => bytes,
        _ => return Json(RpcResponse::error("Invalid transaction data")),
    };

    // Generate transaction signature (hash of transaction)
    let mut hasher = Sha256::new();
    hasher.update(&tx_bytes);
    hasher.update(&std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .to_le_bytes());
    let signature = hasher.finalize().to_vec();

    // Store transaction in state
    let tx_info = TransactionInfo {
        signature: signature.clone(),
        slot: state.chain_state.read().await.current_slot,
        transaction: tx_bytes,
        status: TransactionStatus::Pending,
        fee: 5000, // Base fee
        compute_units: 0,
        logs: vec![],
    };

    // Add to transactions map
    {
        let mut txs = state.transactions.write().await;
        txs.insert(signature.clone(), tx_info);
    }

    // Update transaction count
    {
        let mut chain = state.chain_state.write().await;
        chain.transaction_count += 1;
    }

    info!("Transaction submitted: {}", bs58::encode(&signature).into_string());
    
    Json(RpcResponse::success(json!(bs58::encode(&signature).into_string())))
}

pub async fn simulate_transaction(
    _state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    if params.is_none() {
        return Json(RpcResponse::error("Missing transaction parameter"));
    }

    Json(RpcResponse::success(json!({
        "err": null,
        "logs": [
            "Program log: Instruction: Transfer",
            "Program log: Success"
        ],
        "accounts": null,
        "unitsConsumed": 150,
        "returnData": null
    })))
}

pub async fn get_transaction(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    let signature = match params
        .and_then(|p| p.get("signature"))
        .and_then(|s| s.as_str())
    {
        Some(sig) => sig,
        None => return Json(RpcResponse::error("Missing signature parameter")),
    };

    // Decode signature
    let sig_bytes = match bs58::decode(signature).into_vec() {
        Ok(bytes) => bytes,
        Err(_) => return Json(RpcResponse::error("Invalid signature format")),
    };

    // Look up transaction
    let transactions = state.transactions.read().await;
    
    if let Some(tx_info) = transactions.get(&sig_bytes) {
        let status = match &tx_info.status {
            TransactionStatus::Pending => json!({"Pending": null}),
            TransactionStatus::Processed => json!({"Ok": null}),
            TransactionStatus::Confirmed => json!({"Ok": null}),
            TransactionStatus::Finalized => json!({"Ok": null}),
            TransactionStatus::Failed(msg) => json!({"Err": msg}),
        };

        let err = match &tx_info.status {
            TransactionStatus::Failed(msg) => json!(msg),
            _ => json!(null),
        };

        Json(RpcResponse::success(json!({
            "slot": tx_info.slot,
            "transaction": {
                "signatures": [bs58::encode(&tx_info.signature).into_string()],
                "message": {
                    "header": {
                        "numRequiredSignatures": 1,
                        "numReadonlySignedAccounts": 0,
                        "numReadonlyUnsignedAccounts": 1
                    },
                    "accountKeys": [],
                    "recentBlockhash": "blockhash",
                    "instructions": []
                }
            },
            "meta": {
                "err": err,
                "status": status,
                "fee": tx_info.fee,
                "preBalances": [],
                "postBalances": [],
                "logMessages": tx_info.logs,
                "preTokenBalances": [],
                "postTokenBalances": [],
                "rewards": [],
                "loadedAddresses": {
                    "writable": [],
                    "readonly": []
                },
                "computeUnitsConsumed": tx_info.compute_units
            },
            "blockTime": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            "version": "legacy"
        })))
    } else {
        Json(RpcResponse::error(&format!("Transaction not found: {}", signature)))
    }
}

pub async fn get_transaction_count(state: RpcState) -> Json<RpcResponse> {
    let chain = state.chain_state.read().await;
    Json(RpcResponse::success(json!(chain.transaction_count)))
}

pub async fn get_recent_performance_samples(
    _state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    let limit = params
        .and_then(|p| p.get("limit"))
        .and_then(|l| l.as_u64())
        .unwrap_or(720);

    let samples: Vec<_> = (0..limit.min(100)).map(|i| json!({
        "slot": 1000 - i,
        "numTransactions": 5000,
        "numSlots": 1,
        "samplePeriodSecs": 60
    })).collect();

    Json(RpcResponse::success(json!(samples)))
}

pub async fn get_signature_statuses(
    _state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    if params.is_none() {
        return Json(RpcResponse::error("Missing signatures parameter"));
    }

    Json(RpcResponse::success(json!({
        "value": [
            {
                "slot": 100,
                "confirmations": 32,
                "err": null,
                "status": { "Ok": null },
                "confirmationStatus": "finalized"
            }
        ]
    })))
}

pub async fn get_signatures_for_address(
    _state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    if params.is_none() {
        return Json(RpcResponse::error("Missing address parameter"));
    }

    let limit = params
        .as_ref()
        .and_then(|p| p.get("limit"))
        .and_then(|l| l.as_u64())
        .unwrap_or(1000);

    let signatures: Vec<_> = (0..limit.min(100)).map(|i| json!({
        "signature": format!("sig_{}", i),
        "slot": 100 + i,
        "err": null,
        "memo": null,
        "blockTime": 1234567890 + i,
        "confirmationStatus": "finalized"
    })).collect();

    Json(RpcResponse::success(json!(signatures)))
}

// ============================================================================
// Account Methods
// ============================================================================

pub async fn get_balance(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    let address = match params
        .and_then(|p| p.get("address").or(p.get("pubkey")))
        .and_then(|a| a.as_str())
    {
        Some(addr) => addr,
        None => return Json(RpcResponse::error("Missing account parameter")),
    };

    // Decode address
    let address_bytes = match bs58::decode(address).into_vec() {
        Ok(bytes) => bytes,
        Err(_) => return Json(RpcResponse::error("Invalid address format")),
    };

    let chain = state.chain_state.read().await;
    let current_slot = chain.current_slot;
    drop(chain);

    // Look up account in state
    let accounts = state.accounts.read().await;
    let balance = accounts.get(&address_bytes)
        .map(|account| account.lamports)
        .unwrap_or(0);

    Json(RpcResponse::success(json!({
        "context": { "slot": current_slot },
        "value": balance
    })))
}

pub async fn get_account_info(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    let address = match params
        .and_then(|p| p.get("address").or(p.get("pubkey")))
        .and_then(|a| a.as_str())
    {
        Some(addr) => addr,
        None => return Json(RpcResponse::error("Missing account parameter")),
    };

    // Decode address
    let address_bytes = match bs58::decode(address).into_vec() {
        Ok(bytes) => bytes,
        Err(_) => return Json(RpcResponse::error("Invalid address format")),
    };

    let chain = state.chain_state.read().await;
    let current_slot = chain.current_slot;
    drop(chain);

    // Look up account in state
    let accounts = state.accounts.read().await;
    
    if let Some(account) = accounts.get(&address_bytes) {
        let data_base64 = base64::encode(&account.data);
        let owner = bs58::encode(&account.owner).into_string();
        
        Json(RpcResponse::success(json!({
            "context": { "slot": current_slot },
            "value": {
                "lamports": account.lamports,
                "owner": owner,
                "data": [data_base64, "base64"],
                "executable": account.executable,
                "rentEpoch": account.rent_epoch
            }
        })))
    } else {
        // Account not found
        Json(RpcResponse::success(json!({
            "context": { "slot": current_slot },
            "value": null
        })))
    }
}

pub async fn get_multiple_accounts(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    let addresses = match params
        .and_then(|p| p.get("addresses").or(p.get("pubkeys")))
        .and_then(|a| a.as_array())
    {
        Some(addrs) => addrs.clone(),
        None => return Json(RpcResponse::error("Missing accounts parameter")),
    };

    let chain = state.chain_state.read().await;
    let current_slot = chain.current_slot;
    drop(chain);

    let accounts_map = state.accounts.read().await;
    
    let mut results = Vec::new();
    for addr in addresses {
        if let Some(addr_str) = addr.as_str() {
            if let Ok(addr_bytes) = bs58::decode(addr_str).into_vec() {
                if let Some(account) = accounts_map.get(&addr_bytes) {
                    let data_base64 = base64::encode(&account.data);
                    let owner = bs58::encode(&account.owner).into_string();
                    
                    results.push(json!({
                        "lamports": account.lamports,
                        "owner": owner,
                        "data": [data_base64, "base64"],
                        "executable": account.executable,
                        "rentEpoch": account.rent_epoch
                    }));
                } else {
                    results.push(json!(null));
                }
            } else {
                results.push(json!(null));
            }
        } else {
            results.push(json!(null));
        }
    }

    Json(RpcResponse::success(json!({
        "context": { "slot": current_slot },
        "value": results
    })))
}

pub async fn get_program_accounts(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    let program_id = match params
        .and_then(|p| p.get("programId"))
        .and_then(|p| p.as_str())
    {
        Some(id) => id,
        None => return Json(RpcResponse::error("Missing program ID parameter")),
    };

    // Decode program ID
    let program_id_bytes = match bs58::decode(program_id).into_vec() {
        Ok(bytes) => bytes,
        Err(_) => return Json(RpcResponse::error("Invalid program ID format")),
    };

    // Look up accounts owned by this program
    let accounts = state.accounts.read().await;
    
    let program_accounts: Vec<_> = accounts.iter()
        .filter(|(_, account)| account.owner == program_id_bytes)
        .map(|(addr, account)| {
            let data_base64 = base64::encode(&account.data);
            json!({
                "pubkey": bs58::encode(addr).into_string(),
                "account": {
                    "lamports": account.lamports,
                    "owner": bs58::encode(&account.owner).into_string(),
                    "data": [data_base64, "base64"],
                    "executable": account.executable,
                    "rentEpoch": account.rent_epoch
                }
            })
        })
        .collect();

    Json(RpcResponse::success(json!(program_accounts)))
}

pub async fn get_largest_accounts(
    _state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    Json(RpcResponse::success(json!({
        "context": { "slot": 100 },
        "value": [
            {
                "address": "account1",
                "lamports": 999999999999
            },
            {
                "address": "account2",
                "lamports": 888888888888
            }
        ]
    })))
}

pub async fn get_token_accounts_by_owner(
    _state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    if params.is_none() {
        return Json(RpcResponse::error("Missing owner parameter"));
    }

    Json(RpcResponse::success(json!({
        "context": { "slot": 100 },
        "value": [
            {
                "pubkey": "token_account1",
                "account": {
                    "lamports": 2039280,
                    "owner": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
                    "data": ["base64data", "base64"],
                    "executable": false,
                    "rentEpoch": 0
                }
            }
        ]
    })))
}

pub async fn get_token_supply(
    _state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    if params.is_none() {
        return Json(RpcResponse::error("Missing mint parameter"));
    }

    Json(RpcResponse::success(json!({
        "context": { "slot": 100 },
        "value": {
            "amount": "1000000000",
            "decimals": 9,
            "uiAmount": 1.0,
            "uiAmountString": "1.0"
        }
    })))
}

// ============================================================================
// Validator & Staking Methods
// ============================================================================

pub async fn get_vote_accounts(state: RpcState) -> Json<RpcResponse> {
    Json(RpcResponse::success(json!({
        "current": [
            {
                "votePubkey": "validator1",
                "nodePubkey": "node1",
                "activatedStake": 100000000,
                "epochVoteAccount": true,
                "commission": 5,
                "lastVote": 1000,
                "epochCredits": [[1, 100, 0]],
                "rootSlot": 990
            }
        ],
        "delinquent": []
    })))
}

pub async fn get_validators(state: RpcState) -> Json<RpcResponse> {
    Json(RpcResponse::success(json!({
        "current": [
            {
                "identityPubkey": "validator1",
                "voteAccountPubkey": "vote1",
                "stake": 100000000,
                "commission": 5
            }
        ],
        "delinquent": []
    })))
}

pub async fn get_stake_activation(
    _state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    if params.is_none() {
        return Json(RpcResponse::error("Missing stake account parameter"));
    }

    Json(RpcResponse::success(json!({
        "state": "active",
        "active": 100000000,
        "inactive": 0
    })))
}

pub async fn get_stake_minimum_delegation(_state: RpcState) -> Json<RpcResponse> {
    Json(RpcResponse::success(json!(1000000)))
}

pub async fn get_leader_schedule(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    let slot = params
        .and_then(|p| p.get("slot"))
        .and_then(|s| s.as_u64());

    Json(RpcResponse::success(json!({
        "validator1": [0, 1, 2, 3, 4],
        "validator2": [5, 6, 7, 8, 9]
    })))
}

pub async fn get_epoch_info(state: RpcState) -> Json<RpcResponse> {
    let chain = state.chain_state.read().await;
    let epoch = chain.current_slot / 432000;
    
    Json(RpcResponse::success(json!({
        "absoluteSlot": chain.current_slot,
        "blockHeight": chain.current_slot,
        "epoch": epoch,
        "slotIndex": chain.current_slot % 432000,
        "slotsInEpoch": 432000,
        "transactionCount": chain.transaction_count
    })))
}

pub async fn get_epoch_schedule(_state: RpcState) -> Json<RpcResponse> {
    Json(RpcResponse::success(json!({
        "slotsPerEpoch": 432000,
        "leaderScheduleSlotOffset": 432000,
        "warmup": false,
        "firstNormalEpoch": 0,
        "firstNormalSlot": 0
    })))
}

// ============================================================================
// Network & Cluster Methods
// ============================================================================

pub async fn get_cluster_nodes(_state: RpcState) -> Json<RpcResponse> {
    Json(RpcResponse::success(json!([
        {
            "pubkey": "node1",
            "gossip": "127.0.0.1:9900",
            "tpu": "127.0.0.1:9901",
            "rpc": "127.0.0.1:8899",
            "version": "1.18.0",
            "featureSet": 123456789,
            "shredVersion": 1
        }
    ])))
}

pub async fn get_version(_state: RpcState) -> Json<RpcResponse> {
    Json(RpcResponse::success(json!({
        "solana-core": "1.18.0",
        "feature-set": 123456789
    })))
}

pub async fn get_genesis_hash(_state: RpcState) -> Json<RpcResponse> {
    Json(RpcResponse::success(json!("GenesisHashString")))
}

pub async fn get_identity(_state: RpcState) -> Json<RpcResponse> {
    Json(RpcResponse::success(json!({
        "identity": "NodeIdentityPubkey"
    })))
}

pub async fn get_inflation_governor(_state: RpcState) -> Json<RpcResponse> {
    Json(RpcResponse::success(json!({
        "initial": 0.08,
        "terminal": 0.015,
        "taper": 0.15,
        "foundation": 0.05,
        "foundationTerm": 7.0
    })))
}

pub async fn get_inflation_rate(_state: RpcState) -> Json<RpcResponse> {
    Json(RpcResponse::success(json!({
        "total": 0.08,
        "validator": 0.07,
        "foundation": 0.01,
        "epoch": 100
    })))
}

pub async fn get_inflation_reward(
    _state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    if params.is_none() {
        return Json(RpcResponse::error("Missing addresses parameter"));
    }

    Json(RpcResponse::success(json!([
        {
            "epoch": 100,
            "effectiveSlot": 43200,
            "amount": 1000000,
            "postBalance": 1001000000,
            "commission": 5
        }
    ])))
}

pub async fn get_supply(_state: RpcState) -> Json<RpcResponse> {
    Json(RpcResponse::success(json!({
        "context": { "slot": 100 },
        "value": {
            "total": 1000000000000000,
            "circulating": 900000000000000,
            "nonCirculating": 100000000000000,
            "nonCirculatingAccounts": []
        }
    })))
}

// ============================================================================
// AI Agent Methods (Enhanced)
// ============================================================================

pub async fn deploy_ai_agent(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    let params = match params {
        Some(p) => p,
        None => return Json(RpcResponse::error("Missing deployment parameters")),
    };

    // Extract deployment parameters
    let model_hash = params.get("modelHash")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown");
    
    let owner = params.get("owner")
        .and_then(|o| o.as_str())
        .unwrap_or("unknown");

    let config = params.get("config")
        .cloned()
        .unwrap_or(json!({}));

    // Generate agent ID
    let mut hasher = Sha256::new();
    hasher.update(model_hash.as_bytes());
    hasher.update(owner.as_bytes());
    hasher.update(&std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .to_le_bytes());
    let agent_id = hasher.finalize().to_vec();

    // Create AI agent data
    let agent_data = AIAgentData {
        agent_id: agent_id.clone(),
        model_hash: model_hash.to_string(),
        owner: bs58::decode(owner).into_vec().unwrap_or_else(|_| owner.as_bytes().to_vec()),
        execution_count: 0,
        total_compute_used: 0,
        status: "active".to_string(),
        config: config.clone(),
        created_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };

    // Store agent
    {
        let mut agents = state.ai_agents.write().await;
        agents.insert(agent_id.clone(), agent_data);
    }

    // Generate signature
    let mut sig_hasher = Sha256::new();
    sig_hasher.update(&agent_id);
    let signature = sig_hasher.finalize().to_vec();

    // Update transaction count
    {
        let mut chain = state.chain_state.write().await;
        chain.transaction_count += 1;
    }

    info!("AI Agent deployed: {}", bs58::encode(&agent_id).into_string());

    Json(RpcResponse::success(json!({
        "agentId": bs58::encode(&agent_id).into_string(),
        "signature": bs58::encode(&signature).into_string()
    })))
}

pub async fn execute_ai_agent(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    let params = match params {
        Some(p) => p,
        None => return Json(RpcResponse::error("Missing execution parameters")),
    };

    let agent_id_str = match params.get("agentId").and_then(|a| a.as_str()) {
        Some(id) => id,
        None => return Json(RpcResponse::error("Missing agentId parameter")),
    };

    let agent_id_bytes = match bs58::decode(agent_id_str).into_vec() {
        Ok(bytes) => bytes,
        Err(_) => return Json(RpcResponse::error("Invalid agent ID format")),
    };

    // Look up agent
    let agents = state.ai_agents.read().await;
    
    if let Some(agent) = agents.get(&agent_id_bytes) {
        if agent.status != "active" {
            return Json(RpcResponse::error(&format!("Agent is not active: {}", agent.status)));
        }

        // Get input data
        let _input = params.get("input").cloned().unwrap_or(json!({}));
        
        // Simulate execution (in production, this would call the AI runtime)
        let compute_units = 50000u64; // Base compute cost
        
        // Generate execution ID
        let mut hasher = Sha256::new();
        hasher.update(&agent_id_bytes);
        hasher.update(&std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .to_le_bytes());
        let execution_id = hasher.finalize().to_vec();

        drop(agents);

        // Update agent stats
        {
            let mut agents = state.ai_agents.write().await;
            if let Some(agent) = agents.get_mut(&agent_id_bytes) {
                agent.execution_count += 1;
                agent.total_compute_used += compute_units;
            }
        }

        // Generate signature
        let mut sig_hasher = Sha256::new();
        sig_hasher.update(&execution_id);
        let signature = sig_hasher.finalize().to_vec();

        // Update transaction count
        {
            let mut chain = state.chain_state.write().await;
            chain.transaction_count += 1;
        }

        Json(RpcResponse::success(json!({
            "executionId": bs58::encode(&execution_id).into_string(),
            "signature": bs58::encode(&signature).into_string(),
            "computeUnits": compute_units,
            "output": {
                "result": "AI execution completed",
                "status": "success"
            }
        })))
    } else {
        Json(RpcResponse::error(&format!("Agent not found: {}", agent_id_str)))
    }
}

pub async fn get_ai_agent_info(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    let agent_id_str = match params
        .and_then(|p| p.get("agentId"))
        .and_then(|a| a.as_str())
    {
        Some(id) => id,
        None => return Json(RpcResponse::error("Missing agent ID")),
    };

    let agent_id_bytes = match bs58::decode(agent_id_str).into_vec() {
        Ok(bytes) => bytes,
        Err(_) => return Json(RpcResponse::error("Invalid agent ID format")),
    };

    // Look up agent
    let agents = state.ai_agents.read().await;
    
    if let Some(agent) = agents.get(&agent_id_bytes) {
        Json(RpcResponse::success(json!({
            "agentId": bs58::encode(&agent.agent_id).into_string(),
            "modelHash": agent.model_hash,
            "owner": bs58::encode(&agent.owner).into_string(),
            "executionCount": agent.execution_count,
            "totalComputeUsed": agent.total_compute_used,
            "status": agent.status,
            "createdAt": agent.created_at,
            "config": agent.config
        })))
    } else {
        Json(RpcResponse::error(&format!("Agent not found: {}", agent_id_str)))
    }
}

pub async fn update_ai_agent(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    if params.is_none() {
        return Json(RpcResponse::error("Missing update parameters"));
    }

    let mut chain = state.chain_state.write().await;
    chain.transaction_count += 1;

    Json(RpcResponse::success(json!({
        "signature": format!("sig_{}", chain.transaction_count)
    })))
}

pub async fn list_ai_agents(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    let owner_filter = params
        .and_then(|p| p.get("owner"))
        .and_then(|o| o.as_str())
        .map(|s| bs58::decode(s).into_vec().unwrap_or_else(|_| s.as_bytes().to_vec()));

    let agents = state.ai_agents.read().await;
    
    let mut agent_list: Vec<_> = agents.values()
        .filter(|agent| {
            if let Some(ref owner_bytes) = owner_filter {
                &agent.owner == owner_bytes
            } else {
                true
            }
        })
        .map(|agent| json!({
            "agentId": bs58::encode(&agent.agent_id).into_string(),
            "owner": bs58::encode(&agent.owner).into_string(),
            "status": agent.status,
            "modelHash": agent.model_hash,
            "executionCount": agent.execution_count,
            "createdAt": agent.created_at
        }))
        .collect();

    let total = agent_list.len();

    Json(RpcResponse::success(json!({
        "agents": agent_list,
        "total": total
    })))
}

pub async fn get_ai_agent_executions(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    let agent_id_str = match params
        .and_then(|p| p.get("agentId"))
        .and_then(|a| a.as_str())
    {
        Some(id) => id,
        None => return Json(RpcResponse::error("Missing agent ID")),
    };

    let agent_id_bytes = match bs58::decode(agent_id_str).into_vec() {
        Ok(bytes) => bytes,
        Err(_) => return Json(RpcResponse::error("Invalid agent ID format")),
    };

    // Look up agent
    let agents = state.ai_agents.read().await;
    
    if let Some(agent) = agents.get(&agent_id_bytes) {
        // In production, executions would be stored separately
        // For now, return summary based on agent stats
        let execution_count = agent.execution_count;
        let total_compute = agent.total_compute_used;
        let avg_compute = if execution_count > 0 {
            total_compute / execution_count
        } else {
            0
        };

        // Generate mock execution history based on actual execution count
        let executions: Vec<_> = (0..execution_count.min(10)).map(|i| {
            json!({
                "executionId": format!("exec_{}_{}", bs58::encode(&agent_id_bytes).into_string(), i),
                "timestamp": agent.created_at + (i * 60),
                "computeUnits": avg_compute,
                "status": "success"
            })
        }).collect();

        Json(RpcResponse::success(json!({
            "executions": executions,
            "total": execution_count,
            "totalComputeUsed": total_compute,
            "averageComputePerExecution": avg_compute
        })))
    } else {
        Json(RpcResponse::error(&format!("Agent not found: {}", agent_id_str)))
    }
}

// ============================================================================
// Bridge Methods (Enhanced)
// ============================================================================

/// Bridge operation storage for tracking deposits/withdrawals
#[derive(Debug, Clone)]
pub struct BridgeOperation {
    pub id: Vec<u8>,
    pub op_type: String,
    pub status: String,
    pub from_chain: String,
    pub to_chain: String,
    pub amount: u64,
    pub sender: Vec<u8>,
    pub recipient: Vec<u8>,
    pub confirmations: u32,
    pub timestamp: u64,
}

pub async fn bridge_deposit(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    let params = match params {
        Some(p) => p,
        None => return Json(RpcResponse::error("Missing deposit parameters")),
    };

    // Extract deposit parameters
    let amount = params.get("amount")
        .and_then(|a| a.as_u64())
        .unwrap_or(0);

    if amount == 0 {
        return Json(RpcResponse::error("Invalid deposit amount"));
    }

    let sender = params.get("sender")
        .and_then(|s| s.as_str())
        .unwrap_or("unknown");
    
    let recipient = params.get("recipient")
        .and_then(|r| r.as_str())
        .unwrap_or("unknown");

    let from_chain = params.get("fromChain")
        .and_then(|c| c.as_str())
        .unwrap_or("solana");

    let to_chain = params.get("toChain")
        .and_then(|c| c.as_str())
        .unwrap_or("ade");

    // Generate deposit ID
    let mut hasher = Sha256::new();
    hasher.update(b"deposit");
    hasher.update(sender.as_bytes());
    hasher.update(recipient.as_bytes());
    hasher.update(&amount.to_le_bytes());
    hasher.update(&std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .to_le_bytes());
    let deposit_id = hasher.finalize().to_vec();

    // Generate signature
    let mut sig_hasher = Sha256::new();
    sig_hasher.update(&deposit_id);
    let signature = sig_hasher.finalize().to_vec();

    // Update transaction count and chain state
    {
        let mut chain = state.chain_state.write().await;
        chain.transaction_count += 1;
    }

    info!("Bridge deposit initiated: {} lamports from {} to {}", 
        amount, from_chain, to_chain);

    Json(RpcResponse::success(json!({
        "depositId": bs58::encode(&deposit_id).into_string(),
        "signature": bs58::encode(&signature).into_string(),
        "status": "pending",
        "amount": amount,
        "fromChain": from_chain,
        "toChain": to_chain,
        "estimatedConfirmations": 32,
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    })))
}

pub async fn bridge_withdraw(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    let params = match params {
        Some(p) => p,
        None => return Json(RpcResponse::error("Missing withdrawal parameters")),
    };

    // Extract withdrawal parameters
    let amount = params.get("amount")
        .and_then(|a| a.as_u64())
        .unwrap_or(0);

    if amount == 0 {
        return Json(RpcResponse::error("Invalid withdrawal amount"));
    }

    let sender = params.get("sender")
        .and_then(|s| s.as_str())
        .unwrap_or("unknown");
    
    let recipient = params.get("recipient")
        .and_then(|r| r.as_str())
        .unwrap_or("unknown");

    // Check sender balance
    let sender_bytes = bs58::decode(sender).into_vec().unwrap_or_default();
    {
        let accounts = state.accounts.read().await;
        if let Some(account) = accounts.get(&sender_bytes) {
            if account.lamports < amount {
                return Json(RpcResponse::error("Insufficient balance for withdrawal"));
            }
        } else {
            return Json(RpcResponse::error("Sender account not found"));
        }
    }

    // Generate withdrawal ID
    let mut hasher = Sha256::new();
    hasher.update(b"withdrawal");
    hasher.update(sender.as_bytes());
    hasher.update(recipient.as_bytes());
    hasher.update(&amount.to_le_bytes());
    hasher.update(&std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .to_le_bytes());
    let withdrawal_id = hasher.finalize().to_vec();

    // Generate signature
    let mut sig_hasher = Sha256::new();
    sig_hasher.update(&withdrawal_id);
    let signature = sig_hasher.finalize().to_vec();

    // Deduct from sender (lock tokens)
    {
        let mut accounts = state.accounts.write().await;
        if let Some(account) = accounts.get_mut(&sender_bytes) {
            account.lamports = account.lamports.saturating_sub(amount);
        }
    }

    // Update transaction count
    {
        let mut chain = state.chain_state.write().await;
        chain.transaction_count += 1;
    }

    info!("Bridge withdrawal initiated: {} lamports from ade to solana", amount);

    Json(RpcResponse::success(json!({
        "withdrawalId": bs58::encode(&withdrawal_id).into_string(),
        "signature": bs58::encode(&signature).into_string(),
        "status": "pending",
        "amount": amount,
        "fromChain": "ade",
        "toChain": "solana",
        "estimatedConfirmations": 32,
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    })))
}

pub async fn get_bridge_status(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    let operation_id = match params
        .and_then(|p| p.get("operationId").or(p.get("id")))
        .and_then(|o| o.as_str())
    {
        Some(id) => id,
        None => return Json(RpcResponse::error("Missing operation ID")),
    };

    // Look up in transactions (bridge operations are stored as transactions)
    let chain = state.chain_state.read().await;
    let current_slot = chain.current_slot;
    let finalized_slot = chain.finalized_slot;
    drop(chain);

    // Determine confirmation status based on slots
    // In production, this would look up actual bridge operation state
    let confirmations = if current_slot > finalized_slot {
        (current_slot - finalized_slot).min(32) as u32
    } else {
        32
    };

    let status = if confirmations >= 32 {
        "completed"
    } else if confirmations >= 16 {
        "confirmed"
    } else {
        "pending"
    };

    Json(RpcResponse::success(json!({
        "id": operation_id,
        "type": "bridge_operation",
        "status": status,
        "confirmations": confirmations,
        "fromChain": "solana",
        "toChain": "ade",
        "currentSlot": current_slot,
        "finalizedSlot": finalized_slot,
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    })))
}

pub async fn get_bridge_history(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    let limit = params
        .as_ref()
        .and_then(|p| p.get("limit"))
        .and_then(|l| l.as_u64())
        .unwrap_or(50) as usize;

    let address = params
        .and_then(|p| p.get("address"))
        .and_then(|a| a.as_str());

    // Look up transactions related to bridge operations
    let transactions = state.transactions.read().await;
    
    let mut operations = Vec::new();
    for (_sig, tx_info) in transactions.iter().take(limit) {
        // Filter bridge-related transactions based on logs or other markers
        // In production, this would filter based on transaction type
        if tx_info.logs.iter().any(|log| log.contains("bridge") || log.contains("deposit") || log.contains("withdraw")) {
            let status = match &tx_info.status {
                TransactionStatus::Finalized => "completed",
                TransactionStatus::Confirmed => "confirmed",
                TransactionStatus::Processed => "processing",
                TransactionStatus::Pending => "pending",
                TransactionStatus::Failed(_) => "failed",
            };

            operations.push(json!({
                "id": bs58::encode(&tx_info.signature).into_string(),
                "type": "bridge_operation",
                "status": status,
                "slot": tx_info.slot,
                "fee": tx_info.fee
            }));
        }
    }

    let total = operations.len();

    Json(RpcResponse::success(json!({
        "operations": operations,
        "total": total
    })))
}

pub async fn estimate_bridge_fee(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    let params = match params {
        Some(p) => p,
        None => return Json(RpcResponse::error("Missing fee parameters")),
    };

    let amount = params.get("amount")
        .and_then(|a| a.as_u64())
        .unwrap_or(0);

    // Dynamic fee calculation based on network state
    let chain = state.chain_state.read().await;
    let current_slot = chain.current_slot;
    let tx_count = chain.transaction_count;
    drop(chain);

    // Base fee calculation: minimum fee + congestion premium
    let base_fee = 5000u64; // 5000 lamports base
    
    // Congestion factor based on recent transaction count
    let congestion_factor = if tx_count > 1000 {
        1.5
    } else if tx_count > 500 {
        1.2
    } else {
        1.0
    };

    // Percentage fee (0.1% of amount)
    let percentage_fee_rate = 0.001;
    let percentage_fee = ((amount as f64) * percentage_fee_rate) as u64;

    // Total fee
    let total_fee = ((base_fee as f64 * congestion_factor) as u64) + percentage_fee;

    // Estimated time based on network conditions (in seconds)
    let estimated_time = if tx_count > 1000 {
        120 // 2 minutes during high congestion
    } else if tx_count > 500 {
        90
    } else {
        60 // 1 minute normal
    };

    Json(RpcResponse::success(json!({
        "baseFee": base_fee,
        "percentageFee": percentage_fee,
        "percentageFeeRate": percentage_fee_rate,
        "congestionFactor": congestion_factor,
        "totalFee": total_fee,
        "estimatedTime": estimated_time,
        "currentSlot": current_slot,
        "networkLoad": if tx_count > 1000 { "high" } else if tx_count > 500 { "medium" } else { "low" }
    })))
}

// ============================================================================
// Program & Smart Contract Methods
// ============================================================================

pub async fn request_airdrop(
    _state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    if params.is_none() {
        return Json(RpcResponse::error("Missing airdrop parameters"));
    }

    Json(RpcResponse::success(json!("airdrop_signature")))
}

pub async fn minimum_ledger_slot(_state: RpcState) -> Json<RpcResponse> {
    Json(RpcResponse::success(json!(0)))
}

pub async fn get_slot_leaders(
    _state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    let start = params
        .as_ref()
        .and_then(|p| p.get("start"))
        .and_then(|s| s.as_u64())
        .unwrap_or(0);
    
    let limit = params
        .and_then(|p| p.get("limit"))
        .and_then(|l| l.as_u64())
        .unwrap_or(100);

    let leaders: Vec<_> = (0..limit).map(|i| format!("validator_{}", i % 3)).collect();
    
    Json(RpcResponse::success(json!(leaders)))
}

pub async fn get_fee_for_message(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    if params.is_none() {
        return Json(RpcResponse::error("Missing message parameter"));
    }

    let chain = state.chain_state.read().await;
    let current_slot = chain.current_slot;
    let tx_count = chain.transaction_count;
    drop(chain);

    // Dynamic fee calculation based on network congestion
    let base_fee = 5000u64;
    
    // Congestion multiplier
    let fee = if tx_count > 1000 {
        base_fee * 3 // High congestion
    } else if tx_count > 500 {
        base_fee * 2 // Medium congestion
    } else {
        base_fee // Normal
    };

    Json(RpcResponse::success(json!({
        "context": { "slot": current_slot },
        "value": fee
    })))
}

pub async fn get_recent_prioritization_fees(
    state: RpcState,
    params: Option<serde_json::Value>,
) -> Json<RpcResponse> {
    let chain = state.chain_state.read().await;
    let current_slot = chain.current_slot;
    let tx_count = chain.transaction_count;
    drop(chain);

    // Generate prioritization fee history based on recent slots
    let num_samples = 150.min(current_slot as usize);
    
    let fees: Vec<_> = (0..num_samples).map(|i| {
        let slot = current_slot - i as u64;
        
        // Calculate dynamic fee based on slot distance and network state
        // More recent slots have higher fees during congestion
        let base_priority = if tx_count > 1000 {
            5000u64
        } else if tx_count > 500 {
            1000
        } else {
            100
        };
        
        // Add some variation based on slot
        let fee = base_priority + ((slot % 10) * 100);
        
        json!({
            "slot": slot,
            "prioritizationFee": fee
        })
    }).collect();

    Json(RpcResponse::success(json!(fees)))
}

pub async fn get_max_retransmit_slot(_state: RpcState) -> Json<RpcResponse> {
    Json(RpcResponse::success(json!(1000)))
}

pub async fn get_max_shred_insert_slot(_state: RpcState) -> Json<RpcResponse> {
    Json(RpcResponse::success(json!(1000)))
}
