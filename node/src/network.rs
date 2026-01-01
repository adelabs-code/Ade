use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use tracing::{info, warn, debug};
use std::collections::{HashMap, HashSet};

pub struct NetworkManager {
    gossip_port: u16,
    bootstrap_nodes: Vec<String>,
    peers: Arc<RwLock<HashMap<Vec<u8>, PeerInfo>>>,
    message_cache: Arc<RwLock<HashSet<Vec<u8>>>>,
    max_peers: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub pubkey: Vec<u8>,
    pub address: String,
    pub last_seen: u64,
    pub stake: u64,
    pub version: String,
    pub latency_ms: u64,
    pub is_validator: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GossipMessage {
    /// Peer announcement
    PeerInfo {
        pubkey: Vec<u8>,
        address: String,
        stake: u64,
        version: String,
    },
    /// Block proposal
    BlockProposal {
        slot: u64,
        block_hash: Vec<u8>,
        validator: Vec<u8>,
        block_data: Vec<u8>,
    },
    /// Transaction broadcast
    TransactionBatch {
        transactions: Vec<Vec<u8>>,
    },
    /// Vote broadcast
    Vote {
        slot: u64,
        block_hash: Vec<u8>,
        validator: Vec<u8>,
        signature: Vec<u8>,
    },
    /// Ping/Pong for health check
    Ping {
        timestamp: u64,
    },
    Pong {
        timestamp: u64,
        original_timestamp: u64,
    },
}

impl NetworkManager {
    pub fn new(gossip_port: u16, bootstrap_nodes: Vec<String>) -> Result<Self> {
        Ok(Self {
            gossip_port,
            bootstrap_nodes,
            peers: Arc::new(RwLock::new(HashMap::new())),
            message_cache: Arc::new(RwLock::new(HashSet::new())),
            max_peers: 1000,
        })
    }

    pub async fn start(&self) -> Result<()> {
        info!("Starting network manager on port {}", self.gossip_port);
        
        // Connect to bootstrap nodes
        for node in &self.bootstrap_nodes {
            info!("Connecting to bootstrap node: {}", node);
            self.connect_to_peer(node).await?;
        }

        // Spawn background tasks
        self.spawn_peer_maintenance_task();
        self.spawn_message_processor_task();
        self.spawn_health_check_task();

        Ok(())
    }

    async fn connect_to_peer(&self, address: &str) -> Result<()> {
        let peer = PeerInfo {
            pubkey: vec![0u8; 32], // Would be exchanged during handshake
            address: address.to_string(),
            last_seen: current_timestamp(),
            stake: 0,
            version: "1.0.0".to_string(),
            latency_ms: 0,
            is_validator: false,
        };
        
        let mut peers = self.peers.write().await;
        peers.insert(peer.pubkey.clone(), peer);
        Ok(())
    }

    /// Broadcast a message to all peers
    pub async fn broadcast(&self, message: GossipMessage) -> Result<usize> {
        let message_hash = self.hash_message(&message);
        
        // Check if already processed
        {
            let mut cache = self.message_cache.write().await;
            if cache.contains(&message_hash) {
                return Ok(0);
            }
            cache.insert(message_hash);
            
            // Keep cache size bounded
            if cache.len() > 10000 {
                cache.clear();
            }
        }

        let peers = self.peers.read().await;
        let peer_count = peers.len();
        
        debug!("Broadcasting message to {} peers", peer_count);
        
        // In production, this would send over UDP/TCP
        for (_, peer) in peers.iter() {
            debug!("Sending to peer: {}", peer.address);
        }

        Ok(peer_count)
    }

    /// Broadcast transaction to network
    pub async fn broadcast_transaction(&self, tx_data: &[u8]) -> Result<()> {
        let message = GossipMessage::TransactionBatch {
            transactions: vec![tx_data.to_vec()],
        };
        
        let sent = self.broadcast(message).await?;
        info!("Broadcasted transaction to {} peers", sent);
        
        Ok(())
    }

    /// Broadcast block to network
    pub async fn broadcast_block(&self, slot: u64, block_hash: Vec<u8>, block_data: Vec<u8>, validator: Vec<u8>) -> Result<()> {
        let message = GossipMessage::BlockProposal {
            slot,
            block_hash,
            validator,
            block_data,
        };
        
        let sent = self.broadcast(message).await?;
        info!("Broadcasted block at slot {} to {} peers", slot, sent);
        
        Ok(())
    }

    /// Broadcast vote to network
    pub async fn broadcast_vote(&self, slot: u64, block_hash: Vec<u8>, validator: Vec<u8>, signature: Vec<u8>) -> Result<()> {
        let message = GossipMessage::Vote {
            slot,
            block_hash,
            validator,
            signature,
        };
        
        self.broadcast(message).await?;
        Ok(())
    }

    /// Get peer count
    pub async fn get_peer_count(&self) -> usize {
        self.peers.read().await.len()
    }

    /// Get connected peers
    pub async fn get_peers(&self) -> Vec<PeerInfo> {
        self.peers.read().await.values().cloned().collect()
    }

    /// Get validator peers only
    pub async fn get_validator_peers(&self) -> Vec<PeerInfo> {
        self.peers.read().await
            .values()
            .filter(|p| p.is_validator)
            .cloned()
            .collect()
    }

    /// Remove inactive peers
    pub async fn prune_inactive_peers(&self, timeout_secs: u64) -> usize {
        let cutoff = current_timestamp() - timeout_secs;
        let mut peers = self.peers.write().await;
        
        let initial_count = peers.len();
        peers.retain(|_, peer| peer.last_seen > cutoff);
        let removed = initial_count - peers.len();
        
        if removed > 0 {
            info!("Pruned {} inactive peers", removed);
        }
        
        removed
    }

    /// Update peer info
    pub async fn update_peer(&self, pubkey: &[u8], update_fn: impl FnOnce(&mut PeerInfo)) -> Result<()> {
        let mut peers = self.peers.write().await;
        
        if let Some(peer) = peers.get_mut(pubkey) {
            update_fn(peer);
            peer.last_seen = current_timestamp();
        }
        
        Ok(())
    }

    /// Hash message for deduplication
    fn hash_message(&self, message: &GossipMessage) -> Vec<u8> {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        let serialized = bincode::serialize(message).unwrap_or_default();
        hasher.update(&serialized);
        hasher.finalize().to_vec()
    }

    /// Spawn peer maintenance task
    fn spawn_peer_maintenance_task(&self) {
        let peers = self.peers.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                let peer_count = peers.read().await.len();
                debug!("Peer maintenance: {} peers connected", peer_count);
            }
        });
    }

    /// Spawn message processor task
    fn spawn_message_processor_task(&self) {
        tokio::spawn(async move {
            // In production, this would listen for incoming messages
            loop {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        });
    }

    /// Spawn health check task
    fn spawn_health_check_task(&self) {
        let peers = self.peers.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                
                // Send ping to all peers
                debug!("Sending health check pings");
            }
        });
    }

    /// Get network statistics
    pub async fn get_stats(&self) -> NetworkStats {
        let peers = self.peers.read().await;
        
        let validator_count = peers.values().filter(|p| p.is_validator).count();
        let total_stake: u64 = peers.values().map(|p| p.stake).sum();
        let avg_latency = if !peers.is_empty() {
            peers.values().map(|p| p.latency_ms).sum::<u64>() / peers.len() as u64
        } else {
            0
        };

        NetworkStats {
            connected_peers: peers.len(),
            validator_peers: validator_count,
            total_peer_stake: total_stake,
            average_latency_ms: avg_latency,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    pub connected_peers: usize,
    pub validator_peers: usize,
    pub total_peer_stake: u64,
    pub average_latency_ms: u64,
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

    #[tokio::test]
    async fn test_network_initialization() {
        let network = NetworkManager::new(9900, vec![]).unwrap();
        assert_eq!(network.get_peer_count().await, 0);
    }

    #[tokio::test]
    async fn test_peer_connection() {
        let network = NetworkManager::new(9900, vec![]).unwrap();
        network.connect_to_peer("127.0.0.1:9901").await.unwrap();
        
        assert_eq!(network.get_peer_count().await, 1);
    }

    #[tokio::test]
    async fn test_broadcast() {
        let network = NetworkManager::new(9900, vec![]).unwrap();
        network.connect_to_peer("127.0.0.1:9901").await.unwrap();
        
        let message = GossipMessage::Ping { timestamp: 12345 };
        let sent = network.broadcast(message).await.unwrap();
        
        assert_eq!(sent, 1);
    }

    #[tokio::test]
    async fn test_prune_inactive_peers() {
        let network = NetworkManager::new(9900, vec![]).unwrap();
        network.connect_to_peer("127.0.0.1:9901").await.unwrap();
        
        // Wait a bit
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        // Prune with 0 timeout (should remove all)
        let removed = network.prune_inactive_peers(0).await;
        assert_eq!(removed, 1);
    }
}
