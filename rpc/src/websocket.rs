use axum::{
    extract::{ws::{WebSocket, WebSocketUpgrade, Message}, State},
    response::Response,
};
use tokio::sync::broadcast;
use serde::{Serialize, Deserialize};
use tracing::{info, warn, debug};
use std::sync::Arc;

use crate::state::RpcStateBackend;

/// WebSocket subscription manager
pub struct SubscriptionManager {
    slot_tx: broadcast::Sender<SlotUpdate>,
    account_tx: broadcast::Sender<AccountUpdate>,
    transaction_tx: broadcast::Sender<TransactionUpdate>,
    block_tx: broadcast::Sender<BlockUpdate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotUpdate {
    pub slot: u64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountUpdate {
    pub pubkey: String,
    pub lamports: u64,
    pub slot: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionUpdate {
    pub signature: String,
    pub slot: u64,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockUpdate {
    pub slot: u64,
    pub hash: String,
    pub parent_slot: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", content = "params")]
pub enum SubscriptionRequest {
    SlotSubscribe,
    AccountSubscribe { pubkey: String },
    SignatureSubscribe { signature: String },
    BlockSubscribe,
    Unsubscribe { subscription_id: u64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionResponse {
    pub jsonrpc: String,
    pub method: String,
    pub params: serde_json::Value,
}

impl SubscriptionManager {
    pub fn new() -> Self {
        let (slot_tx, _) = broadcast::channel(1000);
        let (account_tx, _) = broadcast::channel(1000);
        let (transaction_tx, _) = broadcast::channel(1000);
        let (block_tx, _) = broadcast::channel(1000);

        Self {
            slot_tx,
            account_tx,
            transaction_tx,
            block_tx,
        }
    }

    /// Publish slot update
    pub fn publish_slot(&self, update: SlotUpdate) {
        let _ = self.slot_tx.send(update);
    }

    /// Publish account update
    pub fn publish_account(&self, update: AccountUpdate) {
        let _ = self.account_tx.send(update);
    }

    /// Publish transaction update
    pub fn publish_transaction(&self, update: TransactionUpdate) {
        let _ = self.transaction_tx.send(update);
    }

    /// Publish block update
    pub fn publish_block(&self, update: BlockUpdate) {
        let _ = self.block_tx.send(update);
    }

    /// Subscribe to slot updates
    pub fn subscribe_slot(&self) -> broadcast::Receiver<SlotUpdate> {
        self.slot_tx.subscribe()
    }

    /// Subscribe to account updates
    pub fn subscribe_account(&self) -> broadcast::Receiver<AccountUpdate> {
        self.account_tx.subscribe()
    }

    /// Subscribe to transaction updates
    pub fn subscribe_transaction(&self) -> broadcast::Receiver<TransactionUpdate> {
        self.transaction_tx.subscribe()
    }

    /// Subscribe to block updates
    pub fn subscribe_block(&self) -> broadcast::Receiver<BlockUpdate> {
        self.block_tx.subscribe()
    }
}

impl Default for SubscriptionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// WebSocket handler
pub async fn handle_websocket(
    ws: WebSocketUpgrade,
    State(state): State<Arc<SubscriptionManager>>,
) -> Response {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, subscription_manager: Arc<SubscriptionManager>) {
    let (mut sender, mut receiver) = socket.split();

    info!("WebSocket connection established");

    // Handle incoming subscription requests
    let recv_task = tokio::spawn(async move {
        while let Some(msg) = receiver.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    debug!("Received subscription request: {}", text);
                    
                    // Parse subscription request
                    if let Ok(request) = serde_json::from_str::<SubscriptionRequest>(&text) {
                        match request {
                            SubscriptionRequest::SlotSubscribe => {
                                info!("Client subscribed to slots");
                            }
                            SubscriptionRequest::AccountSubscribe { pubkey } => {
                                info!("Client subscribed to account: {}", pubkey);
                            }
                            SubscriptionRequest::SignatureSubscribe { signature } => {
                                info!("Client subscribed to signature: {}", signature);
                            }
                            SubscriptionRequest::BlockSubscribe => {
                                info!("Client subscribed to blocks");
                            }
                            SubscriptionRequest::Unsubscribe { subscription_id } => {
                                info!("Client unsubscribed: {}", subscription_id);
                            }
                        }
                    }
                }
                Ok(Message::Close(_)) => {
                    info!("WebSocket connection closed");
                    break;
                }
                Err(e) => {
                    warn!("WebSocket error: {}", e);
                    break;
                }
                _ => {}
            }
        }
    });

    // Send updates to client
    let send_task = tokio::spawn(async move {
        let mut slot_rx = subscription_manager.subscribe_slot();
        
        loop {
            tokio::select! {
                Ok(slot_update) = slot_rx.recv() => {
                    let response = SubscriptionResponse {
                        jsonrpc: "2.0".to_string(),
                        method: "slotNotification".to_string(),
                        params: serde_json::to_value(&slot_update).unwrap(),
                    };
                    
                    if let Ok(text) = serde_json::to_string(&response) {
                        if sender.send(Message::Text(text)).await.is_err() {
                            break;
                        }
                    }
                }
            }
        }
    });

    // Wait for tasks to complete
    let _ = tokio::join!(recv_task, send_task);
}

/// Slot notifier task
pub async fn spawn_slot_notifier(
    state: Arc<RpcStateBackend>,
    subscription_manager: Arc<SubscriptionManager>,
) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(400));

        loop {
            interval.tick().await;

            let slot = {
                let mut chain = state.chain_state.write().await;
                chain.current_slot += 1;
                chain.current_slot
            };

            subscription_manager.publish_slot(SlotUpdate {
                slot,
                timestamp: current_timestamp(),
            });
        }
    });
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

    #[test]
    fn test_subscription_manager() {
        let manager = SubscriptionManager::new();
        
        let mut rx = manager.subscribe_slot();
        
        manager.publish_slot(SlotUpdate {
            slot: 100,
            timestamp: 12345,
        });
        
        // Try to receive (will timeout in test, but structure is correct)
        assert!(rx.try_recv().is_ok());
    }

    #[tokio::test]
    async fn test_slot_notifier() {
        let state = Arc::new(RpcStateBackend::new());
        let manager = Arc::new(SubscriptionManager::new());
        
        let mut rx = manager.subscribe_slot();
        
        spawn_slot_notifier(state, manager).await;
        
        // Wait for first update
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        
        // Should have received at least one update
        assert!(rx.try_recv().is_ok());
    }
}


