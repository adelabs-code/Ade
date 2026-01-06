use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use tracing::{info, debug, warn};
use std::collections::{HashMap, HashSet, VecDeque};

use crate::network::{PeerInfo, GossipMessage};

/// Peer discovery and management system - PRODUCTION implementation
/// 
/// Handles:
/// - Bootstrap peer connections
/// - Gossip-based peer discovery
/// - Connection pooling and management
/// - Peer scoring and rotation
pub struct PeerDiscovery {
    local_pubkey: Vec<u8>,
    known_peers: Arc<RwLock<HashMap<Vec<u8>, DiscoveredPeer>>>,
    pending_connections: Arc<RwLock<VecDeque<String>>>,
    max_peers: usize,
    discovery_interval_ms: u64,
    /// Bootstrap peers for initial network connection
    bootstrap_peers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredPeer {
    pub pubkey: Vec<u8>,
    pub address: String,
    pub first_seen: u64,
    pub last_seen: u64,
    pub connection_attempts: u32,
    pub successful_connections: u32,
    pub peer_score: f64,
    /// Validator stake (0 for non-validators)
    #[serde(default)]
    pub stake: u64,
    /// Whether this peer is a validator
    #[serde(default)]
    pub is_validator: bool,
}

#[derive(Debug, Clone)]
pub struct PeerScore {
    pub uptime_score: f64,
    pub latency_score: f64,
    pub stake_score: f64,
    pub reliability_score: f64,
    pub total_score: f64,
}

impl PeerDiscovery {
    pub fn new(local_pubkey: Vec<u8>, max_peers: usize) -> Self {
        Self::with_bootstrap(local_pubkey, max_peers, vec![])
    }
    
    /// Create with bootstrap peers for initial network connection
    pub fn with_bootstrap(local_pubkey: Vec<u8>, max_peers: usize, bootstrap_peers: Vec<String>) -> Self {
        Self {
            local_pubkey,
            known_peers: Arc::new(RwLock::new(HashMap::new())),
            pending_connections: Arc::new(RwLock::new(VecDeque::new())),
            max_peers,
            discovery_interval_ms: 10000, // 10 seconds
            bootstrap_peers,
        }
    }
    
    /// Add bootstrap peers dynamically
    pub fn add_bootstrap_peers(&mut self, peers: Vec<String>) {
        self.bootstrap_peers.extend(peers);
    }

    /// Start peer discovery process
    pub async fn start(&self) -> Result<()> {
        info!("Starting peer discovery");
        
        // Spawn discovery task
        self.spawn_discovery_task();
        
        // Spawn connection task
        self.spawn_connection_task();
        
        // Spawn scoring task
        self.spawn_scoring_task();
        
        Ok(())
    }

    /// Add a peer to known peers
    pub async fn add_peer(&self, peer: DiscoveredPeer) -> Result<()> {
        let mut peers = self.known_peers.write().await;
        
        if peers.len() >= self.max_peers {
            // Remove lowest scoring peer
            if let Some(lowest_key) = self.find_lowest_scoring_peer(&peers).await {
                peers.remove(&lowest_key);
            }
        }
        
        peers.insert(peer.pubkey.clone(), peer);
        Ok(())
    }

    /// Process peer info message
    pub async fn process_peer_info(&self, pubkey: Vec<u8>, address: String, stake: u64) -> Result<()> {
        let mut peers = self.known_peers.write().await;
        
        if let Some(peer) = peers.get_mut(&pubkey) {
            // Update existing peer
            peer.last_seen = current_timestamp();
            peer.successful_connections += 1;
        } else {
            // Add new peer
            let discovered_peer = DiscoveredPeer {
                pubkey: pubkey.clone(),
                address: address.clone(),
                first_seen: current_timestamp(),
                last_seen: current_timestamp(),
                connection_attempts: 0,
                successful_connections: 1,
                peer_score: 1.0,
            };
            
            drop(peers);
            self.add_peer(discovered_peer).await?;
            
            // Queue for connection
            let mut pending = self.pending_connections.write().await;
            pending.push_back(address);
        }
        
        Ok(())
    }

    /// Get best peers for connection
    pub async fn get_best_peers(&self, count: usize) -> Vec<DiscoveredPeer> {
        let peers = self.known_peers.read().await;
        
        let mut peer_list: Vec<_> = peers.values().cloned().collect();
        peer_list.sort_by(|a, b| b.peer_score.partial_cmp(&a.peer_score).unwrap());
        
        peer_list.into_iter().take(count).collect()
    }

    /// Calculate peer score
    pub fn calculate_peer_score(&self, peer: &DiscoveredPeer, peer_info: Option<&PeerInfo>) -> PeerScore {
        let now = current_timestamp();
        let age = now.saturating_sub(peer.first_seen);
        
        // Uptime score based on successful connections
        let uptime_score = if peer.connection_attempts > 0 {
            peer.successful_connections as f64 / peer.connection_attempts as f64
        } else {
            0.0
        };
        
        // Latency score (if we have peer info)
        let latency_score = if let Some(info) = peer_info {
            if info.latency_ms == 0 {
                1.0
            } else {
                (1000.0 / info.latency_ms as f64).min(1.0)
            }
        } else {
            0.5
        };
        
        // Stake score
        let stake_score = if let Some(info) = peer_info {
            (info.stake as f64 / 1_000_000_000.0).min(1.0)
        } else {
            0.0
        };
        
        // Reliability score based on last seen
        let time_since_seen = now.saturating_sub(peer.last_seen);
        let reliability_score = if time_since_seen < 60 {
            1.0
        } else if time_since_seen < 300 {
            0.8
        } else if time_since_seen < 3600 {
            0.5
        } else {
            0.2
        };
        
        // Weighted total score
        let total_score = 
            uptime_score * 0.3 +
            latency_score * 0.25 +
            stake_score * 0.25 +
            reliability_score * 0.2;
        
        PeerScore {
            uptime_score,
            latency_score,
            stake_score,
            reliability_score,
            total_score,
        }
    }

    /// Update peer scores
    pub async fn update_peer_scores(&self, active_peers: &HashMap<Vec<u8>, PeerInfo>) {
        let mut peers = self.known_peers.write().await;
        
        for (pubkey, discovered_peer) in peers.iter_mut() {
            let peer_info = active_peers.get(pubkey);
            let score = self.calculate_peer_score(discovered_peer, peer_info);
            discovered_peer.peer_score = score.total_score;
        }
    }

    /// Find lowest scoring peer
    async fn find_lowest_scoring_peer(&self, peers: &HashMap<Vec<u8>, DiscoveredPeer>) -> Option<Vec<u8>> {
        peers.iter()
            .min_by(|a, b| a.1.peer_score.partial_cmp(&b.1.peer_score).unwrap())
            .map(|(k, _)| k.clone())
    }

    /// Spawn discovery task - PRODUCTION implementation
    /// 
    /// Periodically queries connected peers for their peer lists (gossip protocol).
    /// This enables organic network growth and recovery from partitions.
    fn spawn_discovery_task(&self) {
        let known_peers = self.known_peers.clone();
        let pending_connections = self.pending_connections.clone();
        let discovery_interval = self.discovery_interval_ms;
        let max_peers = self.max_peers;
        let bootstrap_peers = self.bootstrap_peers.clone();
        
        tokio::spawn(async move {
            let http_client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new());
            
            let mut interval = tokio::time::interval(
                tokio::time::Duration::from_millis(discovery_interval)
            );
            
            loop {
                interval.tick().await;
                
                let current_peers: Vec<_> = {
                    let peers = known_peers.read().await;
                    peers.values().cloned().collect()
                };
                
                let peer_count = current_peers.len();
                debug!("Discovery tick: {} known peers", peer_count);
                
                // If we have few peers, try bootstrap nodes
                if peer_count < 3 {
                    for bootstrap in &bootstrap_peers {
                        let mut pending = pending_connections.write().await;
                        if !pending.contains(bootstrap) {
                            info!("Adding bootstrap peer to connection queue: {}", bootstrap);
                            pending.push_back(bootstrap.clone());
                        }
                    }
                }
                
                // Request peer lists from connected peers (gossip)
                if peer_count > 0 && peer_count < max_peers {
                    // Pick random peers to query
                    let query_count = (peer_count / 3).max(1).min(5);
                    let peers_to_query: Vec<_> = current_peers.iter()
                        .take(query_count)
                        .collect();
                    
                    for peer in peers_to_query {
                        // Request peer list via HTTP gossip endpoint
                        let url = format!("http://{}/gossip/peers", peer.address);
                        
                        match http_client.get(&url).send().await {
                            Ok(response) => {
                                if let Ok(body) = response.json::<serde_json::Value>().await {
                                    if let Some(peer_list) = body.get("peers").and_then(|p| p.as_array()) {
                                        let mut pending = pending_connections.write().await;
                                        
                                        for peer_info in peer_list {
                                            if let Some(addr) = peer_info.get("address").and_then(|a| a.as_str()) {
                                                // Add to pending if not already known
                                                let known = known_peers.read().await;
                                                let is_known = known.values().any(|p| p.address == addr);
                                                
                                                if !is_known && !pending.contains(&addr.to_string()) {
                                                    debug!("Discovered new peer via gossip: {}", addr);
                                                    pending.push_back(addr.to_string());
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                debug!("Failed to query peer {} for peer list: {}", peer.address, e);
                            }
                        }
                    }
                }
            }
        });
    }

    /// Spawn connection task - PRODUCTION implementation
    /// 
    /// Processes pending connection queue and establishes actual TCP connections.
    /// Performs handshake and adds successfully connected peers to known_peers.
    fn spawn_connection_task(&self) {
        let pending = self.pending_connections.clone();
        let known_peers = self.known_peers.clone();
        let max_peers = self.max_peers;
        
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                
                // Check if we need more peers
                let current_count = known_peers.read().await.len();
                if current_count >= max_peers {
                    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                    continue;
                }
                
                let address = {
                    let mut pending_guard = pending.write().await;
                    pending_guard.pop_front()
                };
                
                if let Some(addr) = address {
                    debug!("Attempting TCP connection to: {}", addr);
                    
                    // Establish actual TCP connection with timeout
                    match tokio::time::timeout(
                        std::time::Duration::from_secs(10),
                        tokio::net::TcpStream::connect(&addr)
                    ).await {
                        Ok(Ok(mut stream)) => {
                            info!("Connected to peer: {}", addr);
                            
                            // Perform handshake
                            use tokio::io::{AsyncReadExt, AsyncWriteExt};
                            
                            // Send handshake message
                            let handshake = serde_json::json!({
                                "type": "handshake",
                                "version": "1.0.0",
                                "timestamp": std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap()
                                    .as_secs()
                            });
                            
                            let handshake_bytes = serde_json::to_vec(&handshake).unwrap_or_default();
                            let len = handshake_bytes.len() as u32;
                            
                            if stream.write_all(&len.to_be_bytes()).await.is_ok() 
                                && stream.write_all(&handshake_bytes).await.is_ok() 
                            {
                                // Read response
                                let mut len_buf = [0u8; 4];
                                if stream.read_exact(&mut len_buf).await.is_ok() {
                                    let response_len = u32::from_be_bytes(len_buf) as usize;
                                    if response_len < 65536 { // Sanity check
                                        let mut response_buf = vec![0u8; response_len];
                                        if stream.read_exact(&mut response_buf).await.is_ok() {
                                            if let Ok(response) = serde_json::from_slice::<serde_json::Value>(&response_buf) {
                                                // Extract peer info from response
                                                let pubkey = response.get("pubkey")
                                                    .and_then(|p| p.as_str())
                                                    .and_then(|p| bs58::decode(p).into_vec().ok())
                                                    .unwrap_or_else(|| {
                                                        // Generate pubkey from address hash
                                                        use sha2::{Sha256, Digest};
                                                        let mut h = Sha256::new();
                                                        h.update(addr.as_bytes());
                                                        h.finalize().to_vec()
                                                    });
                                                
                                                let stake = response.get("stake")
                                                    .and_then(|s| s.as_u64())
                                                    .unwrap_or(0);
                                                
                                                // Add to known peers
                                                let now = std::time::SystemTime::now()
                                                    .duration_since(std::time::UNIX_EPOCH)
                                                    .unwrap()
                                                    .as_secs();
                                                
                                                let discovered_peer = super::peer_discovery::DiscoveredPeer {
                                                    pubkey: pubkey.clone(),
                                                    address: addr.clone(),
                                                    first_seen: now,
                                                    last_seen: now,
                                                    connection_attempts: 1,
                                                    successful_connections: 1,
                                                    peer_score: 1.0,
                                                    stake,
                                                    is_validator: stake > 0,
                                                };
                                                
                                                let mut peers = known_peers.write().await;
                                                peers.insert(pubkey, discovered_peer);
                                                
                                                info!("Successfully added peer: {} (stake: {})", addr, stake);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Ok(Err(e)) => {
                            debug!("Failed to connect to {}: {}", addr, e);
                        }
                        Err(_) => {
                            debug!("Connection timeout for: {}", addr);
                        }
                    }
                }
            }
        });
    }

    /// Spawn scoring task
    fn spawn_scoring_task(&self) {
        let known_peers = self.known_peers.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                debug!("Updating peer scores");
                // Scoring happens in update_peer_scores which needs active peer info
            }
        });
    }

    /// Get discovery statistics
    pub async fn get_stats(&self) -> DiscoveryStats {
        let peers = self.known_peers.read().await;
        let pending = self.pending_connections.read().await;
        
        let high_score_peers = peers.values()
            .filter(|p| p.peer_score > 0.7)
            .count();
        
        DiscoveryStats {
            known_peers: peers.len(),
            pending_connections: pending.len(),
            high_score_peers,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryStats {
    pub known_peers: usize,
    pub pending_connections: usize,
    pub high_score_peers: usize,
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
    fn test_peer_discovery_creation() {
        let discovery = PeerDiscovery::new(vec![1u8; 32], 100);
        assert_eq!(discovery.max_peers, 100);
    }

    #[tokio::test]
    async fn test_add_peer() {
        let discovery = PeerDiscovery::new(vec![1u8; 32], 100);
        
        let peer = DiscoveredPeer {
            pubkey: vec![2u8; 32],
            address: "127.0.0.1:9900".to_string(),
            first_seen: current_timestamp(),
            last_seen: current_timestamp(),
            connection_attempts: 1,
            successful_connections: 1,
            peer_score: 0.8,
        };
        
        discovery.add_peer(peer).await.unwrap();
        
        let peers = discovery.known_peers.read().await;
        assert_eq!(peers.len(), 1);
    }

    #[test]
    fn test_peer_scoring() {
        let discovery = PeerDiscovery::new(vec![1u8; 32], 100);
        
        let peer = DiscoveredPeer {
            pubkey: vec![2u8; 32],
            address: "127.0.0.1:9900".to_string(),
            first_seen: current_timestamp() - 1000,
            last_seen: current_timestamp(),
            connection_attempts: 10,
            successful_connections: 9,
            peer_score: 0.0,
        };
        
        let peer_info = PeerInfo {
            pubkey: vec![2u8; 32],
            address: "127.0.0.1:9900".to_string(),
            last_seen: current_timestamp(),
            stake: 1_000_000_000,
            version: "1.0.0".to_string(),
            latency_ms: 50,
            is_validator: true,
        };
        
        let score = discovery.calculate_peer_score(&peer, Some(&peer_info));
        
        assert!(score.total_score > 0.5);
        assert!(score.uptime_score > 0.8);
    }

    #[tokio::test]
    async fn test_get_best_peers() {
        let discovery = PeerDiscovery::new(vec![1u8; 32], 100);
        
        // Add peers with different scores
        for i in 0..5 {
            let peer = DiscoveredPeer {
                pubkey: vec![i as u8; 32],
                address: format!("127.0.0.1:990{}", i),
                first_seen: current_timestamp(),
                last_seen: current_timestamp(),
                connection_attempts: 1,
                successful_connections: 1,
                peer_score: i as f64 / 10.0,
            };
            discovery.add_peer(peer).await.unwrap();
        }
        
        let best = discovery.get_best_peers(3).await;
        assert_eq!(best.len(), 3);
        
        // Check that peers are sorted by score (descending)
        assert!(best[0].peer_score >= best[1].peer_score);
        assert!(best[1].peer_score >= best[2].peer_score);
    }
}







