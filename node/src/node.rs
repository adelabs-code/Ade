use anyhow::{Result, Context};
use ed25519_dalek::Keypair;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::storage::Storage;
use crate::network::NetworkManager;
use crate::validator::Validator;

#[derive(Debug, Clone)]
pub struct NodeConfig {
    pub rpc_port: u16,
    pub gossip_port: u16,
    pub data_dir: String,
    pub validator_keypair: Option<String>,
    pub bootstrap_nodes: Vec<String>,
    pub validator_mode: bool,
}

pub struct Node {
    config: NodeConfig,
    storage: Arc<Storage>,
    network: Arc<NetworkManager>,
    validator: Option<Arc<Validator>>,
    state: Arc<RwLock<NodeState>>,
}

#[derive(Debug, Clone)]
pub struct NodeState {
    pub current_slot: u64,
    pub latest_blockhash: String,
    pub validator_count: usize,
    pub transaction_count: u64,
}

impl Node {
    pub fn new(config: NodeConfig) -> Result<Self> {
        info!("Initializing node with config: {:?}", config);

        let storage = Arc::new(Storage::new(&config.data_dir)
            .context("Failed to initialize storage")?);

        let network = Arc::new(NetworkManager::new(
            config.gossip_port,
            config.bootstrap_nodes.clone(),
        )?);

        let validator = if config.validator_mode {
            let keypair = load_or_generate_keypair(config.validator_keypair.as_deref())?;
            Some(Arc::new(Validator::new(keypair, storage.clone())?))
        } else {
            None
        };

        let state = Arc::new(RwLock::new(NodeState {
            current_slot: 0,
            latest_blockhash: String::new(),
            validator_count: 0,
            transaction_count: 0,
        }));

        Ok(Self {
            config,
            storage,
            network,
            validator,
            state,
        })
    }

    pub async fn start(self) -> Result<()> {
        info!("Starting node services");

        let network_handle = {
            let network = self.network.clone();
            tokio::spawn(async move {
                if let Err(e) = network.start().await {
                    warn!("Network manager error: {}", e);
                }
            })
        };

        if let Some(validator) = &self.validator {
            let validator = validator.clone();
            tokio::spawn(async move {
                if let Err(e) = validator.start().await {
                    warn!("Validator error: {}", e);
                }
            });
        }

        let slot_updater = {
            let state = self.state.clone();
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(tokio::time::Duration::from_millis(400)).await;
                    let mut state = state.write().await;
                    state.current_slot += 1;
                }
            })
        };

        info!("Node started successfully");
        
        tokio::select! {
            _ = network_handle => info!("Network service stopped"),
            _ = slot_updater => info!("Slot updater stopped"),
            _ = tokio::signal::ctrl_c() => info!("Shutdown signal received"),
        }

        info!("Node shutdown complete");
        Ok(())
    }

    pub async fn get_state(&self) -> NodeState {
        self.state.read().await.clone()
    }
}

fn load_or_generate_keypair(path: Option<&str>) -> Result<Keypair> {
    match path {
        Some(p) => {
            let data = std::fs::read(p)
                .context("Failed to read keypair file")?;
            let keypair = Keypair::from_bytes(&data)
                .map_err(|e| anyhow::anyhow!("Invalid keypair: {}", e))?;
            Ok(keypair)
        }
        None => {
            info!("Generating new validator keypair");
            let mut csprng = rand::rngs::OsRng;
            Ok(Keypair::generate(&mut csprng))
        }
    }
}



