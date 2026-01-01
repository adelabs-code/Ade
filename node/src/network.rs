use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

pub struct NetworkManager {
    gossip_port: u16,
    bootstrap_nodes: Vec<String>,
    peers: Arc<RwLock<Vec<PeerInfo>>>,
}

#[derive(Debug, Clone)]
pub struct PeerInfo {
    pub address: String,
    pub last_seen: u64,
    pub stake: u64,
}

impl NetworkManager {
    pub fn new(gossip_port: u16, bootstrap_nodes: Vec<String>) -> Result<Self> {
        Ok(Self {
            gossip_port,
            bootstrap_nodes,
            peers: Arc::new(RwLock::new(Vec::new())),
        })
    }

    pub async fn start(&self) -> Result<()> {
        info!("Starting network manager on port {}", self.gossip_port);
        
        for node in &self.bootstrap_nodes {
            info!("Connecting to bootstrap node: {}", node);
            self.connect_to_peer(node).await?;
        }

        let peers = self.peers.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;
                let peer_count = peers.read().await.len();
                info!("Connected peers: {}", peer_count);
            }
        });

        Ok(())
    }

    async fn connect_to_peer(&self, address: &str) -> Result<()> {
        let peer = PeerInfo {
            address: address.to_string(),
            last_seen: current_timestamp(),
            stake: 0,
        };
        
        let mut peers = self.peers.write().await;
        peers.push(peer);
        Ok(())
    }

    pub async fn broadcast_transaction(&self, tx_data: &[u8]) -> Result<()> {
        let peers = self.peers.read().await;
        info!("Broadcasting transaction to {} peers", peers.len());
        Ok(())
    }

    pub async fn get_peer_count(&self) -> usize {
        self.peers.read().await.len()
    }
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}


