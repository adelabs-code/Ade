use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use tracing::{info, debug, warn};
use std::collections::{HashMap, HashSet, VecDeque};

use crate::network::{PeerInfo, GossipMessage};

/// Peer discovery and management system
pub struct PeerDiscovery {
    local_pubkey: Vec<u8>,
    known_peers: Arc<RwLock<HashMap<Vec<u8>, DiscoveredPeer>>>,
    pending_connections: Arc<RwLock<VecDeque<String>>>,
    max_peers: usize,
    discovery_interval_ms: u64,
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
        Self {
            local_pubkey,
            known_peers: Arc::new(RwLock::new(HashMap::new())),
            pending_connections: Arc::new(RwLock::new(VecDeque::new())),
            max_peers,
            discovery_interval_ms: 10000, // 10 seconds
        }
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

    /// Spawn discovery task
    fn spawn_discovery_task(&self) {
        let known_peers = self.known_peers.clone();
        let discovery_interval = self.discovery_interval_ms;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                tokio::time::Duration::from_millis(discovery_interval)
            );
            
            loop {
                interval.tick().await;
                
                let peer_count = known_peers.read().await.len();
                debug!("Discovery: {} known peers", peer_count);
                
                // In production: request peer lists from connected peers
            }
        });
    }

    /// Spawn connection task
    fn spawn_connection_task(&self) {
        let pending = self.pending_connections.clone();
        
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                
                let address = {
                    let mut pending_guard = pending.write().await;
                    pending_guard.pop_front()
                };
                
                if let Some(addr) = address {
                    debug!("Attempting to connect to: {}", addr);
                    // In production: establish actual connection
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





