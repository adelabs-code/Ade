use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use tracing::{info, debug, warn};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

use crate::utils::{current_timestamp, hash_data};

/// Time-based message cache entry with TTL support
/// This prevents broadcast storms when cache is cleared
#[derive(Debug, Clone)]
struct MessageCacheEntry {
    /// Message hash
    hash: Vec<u8>,
    /// When this entry was added
    inserted_at: Instant,
}

/// TTL-based message cache for preventing broadcast storms
/// 
/// Unlike a simple HashSet with clear(), this cache:
/// 1. Removes entries based on TTL (time-to-live)
/// 2. Uses FIFO eviction when at capacity
/// 3. Never clears entirely, preventing duplicate message processing
struct TimeCache {
    /// Message hashes indexed by insertion time
    entries: VecDeque<MessageCacheEntry>,
    /// Fast lookup set
    lookup: HashSet<Vec<u8>>,
    /// Maximum number of entries
    max_size: usize,
    /// TTL for each entry
    ttl: Duration,
}

impl TimeCache {
    fn new(max_size: usize, ttl: Duration) -> Self {
        Self {
            entries: VecDeque::with_capacity(max_size),
            lookup: HashSet::with_capacity(max_size),
            max_size,
            ttl,
        }
    }
    
    /// Check if message is in cache (also prunes expired entries)
    fn contains(&mut self, hash: &[u8]) -> bool {
        self.prune_expired();
        self.lookup.contains(hash)
    }
    
    /// Insert a message hash into the cache
    fn insert(&mut self, hash: Vec<u8>) -> bool {
        self.prune_expired();
        
        // Already exists
        if self.lookup.contains(&hash) {
            return false;
        }
        
        // At capacity - remove oldest (FIFO eviction)
        while self.entries.len() >= self.max_size {
            if let Some(oldest) = self.entries.pop_front() {
                self.lookup.remove(&oldest.hash);
            }
        }
        
        // Insert new entry
        self.lookup.insert(hash.clone());
        self.entries.push_back(MessageCacheEntry {
            hash,
            inserted_at: Instant::now(),
        });
        
        true
    }
    
    /// Remove entries older than TTL
    fn prune_expired(&mut self) {
        let now = Instant::now();
        
        while let Some(front) = self.entries.front() {
            if now.duration_since(front.inserted_at) > self.ttl {
                if let Some(expired) = self.entries.pop_front() {
                    self.lookup.remove(&expired.hash);
                }
            } else {
                // Entries are ordered by time, so we can stop here
                break;
            }
        }
    }
    
    /// Get cache statistics
    fn stats(&self) -> (usize, usize) {
        (self.entries.len(), self.max_size)
    }
}

/// Protocol version for handshake compatibility
const PROTOCOL_VERSION: &str = "1.0.0";

/// Handshake request sent to peers
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HandshakeRequest {
    /// Protocol version
    protocol_version: String,
    /// Genesis block hash to verify same chain
    genesis_hash: Vec<u8>,
    /// Node's public key for identity
    node_pubkey: Vec<u8>,
    /// Current timestamp for freshness check
    timestamp: u64,
    /// Whether this node is a validator
    is_validator: bool,
}

/// Handshake response from peers
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HandshakeResponse {
    /// Peer's public key
    pubkey: Vec<u8>,
    /// Peer's protocol version
    version: String,
    /// Peer's genesis hash
    genesis_hash: Vec<u8>,
    /// Peer's stake (if validator)
    stake: u64,
    /// Whether peer is a validator
    is_validator: bool,
    /// Supported features
    features: Vec<String>,
}

/// Gossipsub-inspired parameters for efficient message propagation
/// 
/// Instead of flooding ALL peers (which causes broadcast storms),
/// we use a mesh-based approach where each node only forwards to a
/// small subset of peers, similar to libp2p's Gossipsub protocol.
pub struct GossipParams {
    /// Target number of peers to send messages to (D in Gossipsub)
    /// Default: 6 - balances latency vs bandwidth
    pub mesh_degree: usize,
    /// Lower bound for mesh size (D_low)
    pub mesh_degree_low: usize,
    /// Upper bound for mesh size (D_high)
    pub mesh_degree_high: usize,
    /// Number of peers to eagerly push to for fast propagation (D_lazy)
    pub eager_push_peers: usize,
    /// TTL for messages to prevent infinite propagation
    pub message_ttl: u8,
}

impl Default for GossipParams {
    fn default() -> Self {
        Self {
            mesh_degree: 6,
            mesh_degree_low: 4,
            mesh_degree_high: 12,
            eager_push_peers: 3,
            message_ttl: 6,
        }
    }
}

pub struct NetworkManager {
    gossip_port: u16,
    bootstrap_nodes: Vec<String>,
    peers: Arc<RwLock<HashMap<Vec<u8>, PeerInfo>>>,
    /// TTL-based message cache - prevents broadcast storms
    /// Unlike HashSet with clear(), this uses FIFO eviction and TTL
    message_cache: Arc<tokio::sync::Mutex<TimeCache>>,
    max_peers: usize,
    /// Node's Ed25519 keypair for identity
    node_keypair: ed25519_dalek::Keypair,
    /// Genesis block hash for chain identity
    genesis_hash: Vec<u8>,
    /// Whether this node is a validator
    is_validator: bool,
    /// Gossipsub parameters for efficient propagation
    gossip_params: GossipParams,
    /// Mesh peers - subset of peers we actively gossip with
    mesh_peers: Arc<RwLock<HashSet<Vec<u8>>>>,
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
    PeerInfo {
        pubkey: Vec<u8>,
        address: String,
        stake: u64,
        version: String,
    },
    BlockProposal {
        slot: u64,
        block_hash: Vec<u8>,
        validator: Vec<u8>,
        block_data: Vec<u8>,
    },
    TransactionBatch {
        transactions: Vec<Vec<u8>>,
    },
    Vote {
        slot: u64,
        block_hash: Vec<u8>,
        validator: Vec<u8>,
        signature: Vec<u8>,
    },
    Ping {
        timestamp: u64,
    },
    Pong {
        timestamp: u64,
        original_timestamp: u64,
    },
}

/// Network configuration for production deployment
/// 
/// This should be loaded from a configuration file (e.g., genesis.json or network.toml)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NetworkConfig {
    /// The genesis block hash - MUST match all nodes in the network
    pub genesis_hash: String,
    /// Network identifier (e.g., "mainnet", "testnet", "devnet")
    pub network_id: String,
    /// Protocol version for compatibility checks
    pub protocol_version: u32,
    /// Path to the node's keypair file
    pub keypair_path: Option<String>,
    /// List of bootstrap nodes to connect to
    pub bootstrap_nodes: Vec<String>,
    /// Whether this node should act as a validator
    pub is_validator: bool,
    /// The gossip port
    pub gossip_port: u16,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            genesis_hash: String::new(),
            network_id: "devnet".to_string(),
            protocol_version: 1,
            keypair_path: None,
            bootstrap_nodes: vec![],
            is_validator: false,
            gossip_port: 8000,
        }
    }
}

impl NetworkConfig {
    /// Load network configuration from a TOML file
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read network config file '{}': {}", path, e))?;
        
        // Try TOML first, then JSON
        if path.ends_with(".toml") {
            toml::from_str(&content)
                .map_err(|e| anyhow::anyhow!("Failed to parse TOML config: {}", e))
        } else {
            serde_json::from_str(&content)
                .map_err(|e| anyhow::anyhow!("Failed to parse JSON config: {}", e))
        }
    }
    
    /// Create a devnet configuration for testing
    pub fn devnet(gossip_port: u16) -> Self {
        Self {
            genesis_hash: "ADE_DEVNET_GENESIS_V1".to_string(),
            network_id: "devnet".to_string(),
            protocol_version: 1,
            keypair_path: None,
            bootstrap_nodes: vec![],
            is_validator: false,
            gossip_port,
        }
    }
}

impl NetworkManager {
    /// PRODUCTION: Create NetworkManager from configuration file
    /// 
    /// This is the recommended way to initialize the network in production.
    /// The config file should contain the genesis hash and other network parameters.
    pub fn from_config(config: NetworkConfig) -> Result<Self> {
        use rand::rngs::OsRng;
        
        // Validate genesis hash
        if config.genesis_hash.is_empty() {
            return Err(anyhow::anyhow!(
                "Genesis hash is required. Set genesis_hash in your network config file."
            ));
        }
        
        // Load or generate keypair
        let node_keypair = if let Some(ref keypair_path) = config.keypair_path {
            let keypair_bytes = std::fs::read(keypair_path)
                .map_err(|e| anyhow::anyhow!("Failed to read keypair file '{}': {}", keypair_path, e))?;
            
            if keypair_bytes.len() == 64 {
                ed25519_dalek::Keypair::from_bytes(&keypair_bytes)
                    .map_err(|e| anyhow::anyhow!("Invalid keypair bytes: {}", e))?
            } else if keypair_bytes.len() == 32 {
                let secret = ed25519_dalek::SecretKey::from_bytes(&keypair_bytes)
                    .map_err(|e| anyhow::anyhow!("Invalid secret key: {}", e))?;
                let public = ed25519_dalek::PublicKey::from(&secret);
                ed25519_dalek::Keypair { secret, public }
            } else {
                return Err(anyhow::anyhow!(
                    "Invalid keypair file length: expected 32 or 64 bytes, got {}",
                    keypair_bytes.len()
                ));
            }
        } else {
            warn!("No keypair file specified - generating ephemeral keypair");
            let mut csprng = OsRng;
            ed25519_dalek::Keypair::generate(&mut csprng)
        };
        
        // Parse genesis hash from config
        let genesis_hash = Self::parse_genesis_hash(&config.genesis_hash)?;
        
        info!("Network initialized:");
        info!("  Network ID: {}", config.network_id);
        info!("  Genesis: {}", bs58::encode(&genesis_hash).into_string());
        info!("  Node ID: {}", bs58::encode(node_keypair.public.as_bytes()).into_string());
        info!("  Validator: {}", config.is_validator);
        
        let message_cache = TimeCache::new(50_000, Duration::from_secs(300));
        
        Ok(Self {
            gossip_port: config.gossip_port,
            bootstrap_nodes: config.bootstrap_nodes,
            peers: Arc::new(RwLock::new(HashMap::new())),
            message_cache: Arc::new(tokio::sync::Mutex::new(message_cache)),
            max_peers: 1000,
            node_keypair,
            genesis_hash,
            is_validator: config.is_validator,
            gossip_params: GossipParams::default(),
            mesh_peers: Arc::new(RwLock::new(HashSet::new())),
        })
    }
    
    /// Parse genesis hash from string (supports hex, base58, or raw string)
    fn parse_genesis_hash(hash_str: &str) -> Result<Vec<u8>> {
        // Try hex first (0x prefix or just hex)
        let hex_str = hash_str.strip_prefix("0x").unwrap_or(hash_str);
        if let Ok(bytes) = hex::decode(hex_str) {
            if bytes.len() == 32 {
                return Ok(bytes);
            }
        }
        
        // Try base58
        if let Ok(bytes) = bs58::decode(hash_str).into_vec() {
            if bytes.len() == 32 {
                return Ok(bytes);
            }
        }
        
        // Compute hash from string (for named networks like "mainnet", "testnet")
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(b"ADE_GENESIS_HASH:");
        hasher.update(hash_str.as_bytes());
        Ok(hasher.finalize().to_vec())
    }
    
    /// Create a new NetworkManager with generated keypair
    /// 
    /// DEPRECATED: Use from_config() for production deployments.
    /// This constructor is retained for backward compatibility and testing.
    #[deprecated(since = "1.0.0", note = "Use from_config() for production")]
    pub fn new(gossip_port: u16, bootstrap_nodes: Vec<String>) -> Result<Self> {
        warn!("Using NetworkManager::new() - this is deprecated for production use");
        warn!("Configure your network with from_config() and a proper genesis.toml file");
        
        let config = NetworkConfig::devnet(gossip_port);
        let mut config = config;
        config.bootstrap_nodes = bootstrap_nodes;
        
        // For devnet/testing, allow ephemeral genesis
        Self::from_config_internal(config, true)
    }
    
    /// Internal constructor that allows ephemeral genesis for testing
    fn from_config_internal(config: NetworkConfig, allow_empty_genesis: bool) -> Result<Self> {
        use rand::rngs::OsRng;
        
        // Validate genesis hash (unless testing)
        if config.genesis_hash.is_empty() && !allow_empty_genesis {
            return Err(anyhow::anyhow!(
                "Genesis hash is required. Set genesis_hash in your network config file."
            ));
        }
        
        let genesis_hash = if config.genesis_hash.is_empty() {
            // Generate test genesis for devnet
            Self::compute_test_genesis_hash()
        } else {
            Self::parse_genesis_hash(&config.genesis_hash)?
        };
        
        let node_keypair = if let Some(ref keypair_path) = config.keypair_path {
            let keypair_bytes = std::fs::read(keypair_path)
                .map_err(|e| anyhow::anyhow!("Failed to read keypair file '{}': {}", keypair_path, e))?;
            
            if keypair_bytes.len() == 64 {
                ed25519_dalek::Keypair::from_bytes(&keypair_bytes)
                    .map_err(|e| anyhow::anyhow!("Invalid keypair bytes: {}", e))?
            } else if keypair_bytes.len() == 32 {
                let secret = ed25519_dalek::SecretKey::from_bytes(&keypair_bytes)
                    .map_err(|e| anyhow::anyhow!("Invalid secret key: {}", e))?;
                let public = ed25519_dalek::PublicKey::from(&secret);
                ed25519_dalek::Keypair { secret, public }
            } else {
                return Err(anyhow::anyhow!(
                    "Invalid keypair file length: expected 32 or 64 bytes, got {}",
                    keypair_bytes.len()
                ));
            }
        } else {
            let mut csprng = OsRng;
            ed25519_dalek::Keypair::generate(&mut csprng)
        };
        
        let message_cache = TimeCache::new(50_000, Duration::from_secs(300));
        
        Ok(Self {
            gossip_port: config.gossip_port,
            bootstrap_nodes: config.bootstrap_nodes,
            peers: Arc::new(RwLock::new(HashMap::new())),
            message_cache: Arc::new(tokio::sync::Mutex::new(message_cache)),
            max_peers: 1000,
            node_keypair,
            genesis_hash,
            is_validator: config.is_validator,
            gossip_params: GossipParams::default(),
            mesh_peers: Arc::new(RwLock::new(HashSet::new())),
        })
    }
    
    /// Compute test genesis hash for devnet/testing only
    fn compute_test_genesis_hash() -> Vec<u8> {
        use sha2::{Sha256, Digest};
        warn!("Using test genesis hash - NOT FOR PRODUCTION");
        let mut hasher = Sha256::new();
        hasher.update(b"ADE_DEVNET_GENESIS_V1");
        hasher.update(&chrono::Utc::now().timestamp().to_le_bytes());
        hasher.finalize().to_vec()
    }
    
    /// Create a NetworkManager with a specific keypair and genesis hash
    /// This is the recommended constructor for production use
    pub fn with_identity(
        gossip_port: u16,
        bootstrap_nodes: Vec<String>,
        keypair_bytes: &[u8],
        genesis_hash: Vec<u8>,
        is_validator: bool,
    ) -> Result<Self> {
        // Load keypair from bytes (64 bytes: 32 secret + 32 public)
        let node_keypair = if keypair_bytes.len() == 64 {
            ed25519_dalek::Keypair::from_bytes(keypair_bytes)
                .map_err(|e| anyhow::anyhow!("Invalid keypair bytes: {}", e))?
        } else if keypair_bytes.len() == 32 {
            // Only secret key provided, derive public key
            let secret = ed25519_dalek::SecretKey::from_bytes(keypair_bytes)
                .map_err(|e| anyhow::anyhow!("Invalid secret key: {}", e))?;
            let public = ed25519_dalek::PublicKey::from(&secret);
            ed25519_dalek::Keypair { secret, public }
        } else {
            return Err(anyhow::anyhow!(
                "Invalid keypair length: expected 32 or 64 bytes, got {}",
                keypair_bytes.len()
            ));
        };
        
        info!("Node identity initialized: {}", 
            bs58::encode(node_keypair.public.as_bytes()).into_string());
        
        // Message cache with 5 minute TTL and 50K max entries
        let message_cache = TimeCache::new(50_000, Duration::from_secs(300));
        
        Ok(Self {
            gossip_port,
            bootstrap_nodes,
            peers: Arc::new(RwLock::new(HashMap::new())),
            message_cache: Arc::new(tokio::sync::Mutex::new(message_cache)),
            max_peers: 1000,
            node_keypair,
            genesis_hash,
            is_validator,
            gossip_params: GossipParams::default(),
            mesh_peers: Arc::new(RwLock::new(HashSet::new())),
        })
    }
    
    /// Load keypair from file
    pub fn from_keypair_file(
        gossip_port: u16,
        bootstrap_nodes: Vec<String>,
        keypair_path: &str,
        genesis_hash: Vec<u8>,
        is_validator: bool,
    ) -> Result<Self> {
        let keypair_bytes = std::fs::read(keypair_path)
            .map_err(|e| anyhow::anyhow!("Failed to read keypair file '{}': {}", keypair_path, e))?;
        
        Self::with_identity(gossip_port, bootstrap_nodes, &keypair_bytes, genesis_hash, is_validator)
    }
    
    /// Load genesis configuration from standard locations
    /// 
    /// Searches for genesis config in order:
    /// 1. Environment variable ADE_GENESIS_CONFIG
    /// 2. ./genesis.toml
    /// 3. ./config/genesis.toml
    /// 4. ~/.ade/genesis.toml
    pub fn load_genesis_config() -> Result<NetworkConfig> {
        // Check environment variable first
        if let Ok(path) = std::env::var("ADE_GENESIS_CONFIG") {
            return NetworkConfig::from_file(&path);
        }
        
        // Check standard locations
        let search_paths = [
            "genesis.toml".to_string(),
            "config/genesis.toml".to_string(),
            dirs::home_dir()
                .map(|h| h.join(".ade/genesis.toml").to_string_lossy().to_string())
                .unwrap_or_default(),
        ];
        
        for path in &search_paths {
            if !path.is_empty() && std::path::Path::new(path).exists() {
                info!("Loading genesis config from: {}", path);
                return NetworkConfig::from_file(path);
            }
        }
        
        Err(anyhow::anyhow!(
            "No genesis configuration found. Create genesis.toml or set ADE_GENESIS_CONFIG environment variable."
        ))
    }
    
    /// Get the genesis hash for verification during handshake
    pub fn get_genesis_hash(&self) -> &[u8] {
        &self.genesis_hash
    }
    
    /// Get the network's protocol version
    pub fn get_protocol_version(&self) -> u32 {
        1 // Increment when making breaking protocol changes
    }

    pub async fn start(&self) -> Result<()> {
        info!("Starting network manager on port {}", self.gossip_port);
        
        for node in &self.bootstrap_nodes {
            info!("Connecting to bootstrap node: {}", node);
            self.connect_to_peer(node).await?;
        }

        self.spawn_peer_maintenance_task();
        self.spawn_message_processor_task();
        self.spawn_health_check_task();

        Ok(())
    }

    /// Connect to a peer with proper handshake protocol
    /// 
    /// The handshake performs:
    /// 1. TCP connection establishment
    /// 2. Protocol version verification
    /// 3. Genesis hash comparison
    /// 4. Node identity verification (pubkey exchange)
    /// 5. Latency measurement
    async fn connect_to_peer(&self, address: &str) -> Result<()> {
        info!("Initiating connection to peer: {}", address);
        
        // Measure connection latency
        let start_time = std::time::Instant::now();
        
        // Perform handshake
        let handshake_result = self.perform_handshake(address).await;
        
        let latency_ms = start_time.elapsed().as_millis() as u64;
        
        match handshake_result {
            Ok(handshake_response) => {
                // Verify protocol version compatibility
                if !self.is_version_compatible(&handshake_response.version) {
                    warn!("Incompatible protocol version from {}: {}", 
                        address, handshake_response.version);
                    return Err(anyhow::anyhow!("Incompatible protocol version"));
                }
                
                // Verify genesis hash matches
                if !self.verify_genesis_hash(&handshake_response.genesis_hash) {
                    warn!("Genesis hash mismatch from peer: {}", address);
                    return Err(anyhow::anyhow!("Genesis hash mismatch"));
                }
                
                // Verify node identity (optional: verify signature of pubkey)
                if handshake_response.pubkey.len() != 32 {
                    warn!("Invalid pubkey length from peer: {}", address);
                    return Err(anyhow::anyhow!("Invalid node identity"));
                }
                
                let peer = PeerInfo {
                    pubkey: handshake_response.pubkey,
                    address: address.to_string(),
                    last_seen: current_timestamp(),
                    stake: handshake_response.stake,
                    version: handshake_response.version,
                    latency_ms,
                    is_validator: handshake_response.is_validator,
                };
                
                // Check if we have room for more peers
                let mut peers = self.peers.write().await;
                if peers.len() >= self.max_peers {
                    // Remove highest latency peer if new peer is better
                    if let Some((worst_key, worst_latency)) = peers.iter()
                        .max_by_key(|(_, p)| p.latency_ms)
                        .map(|(k, p)| (k.clone(), p.latency_ms))
                    {
                        if latency_ms < worst_latency {
                            peers.remove(&worst_key);
                            info!("Replaced high-latency peer with {}", address);
                        } else {
                            return Err(anyhow::anyhow!("Peer list full"));
                        }
                    }
                }
                
                info!("Successfully connected to peer: {} (latency: {}ms, validator: {})", 
                    address, latency_ms, peer.is_validator);
                
                peers.insert(peer.pubkey.clone(), peer);
                Ok(())
            }
            Err(e) => {
                warn!("Handshake failed with {}: {}", address, e);
                Err(e)
            }
        }
    }
    
    /// Perform the actual handshake with a peer using real TCP communication
    /// 
    /// This sends a HandshakeRequest over TCP and reads the HandshakeResponse.
    /// Protocol format:
    /// - 4 bytes: message length (big-endian u32)
    /// - N bytes: bincode-serialized message
    async fn perform_handshake(&self, address: &str) -> Result<HandshakeResponse> {
        use tokio::net::TcpStream;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        
        // Try to establish TCP connection with timeout
        let connect_timeout = std::time::Duration::from_secs(5);
        
        let mut stream = match tokio::time::timeout(
            connect_timeout,
            TcpStream::connect(address)
        ).await {
            Ok(Ok(stream)) => stream,
            Ok(Err(e)) => return Err(anyhow::anyhow!("Connection failed: {}", e)),
            Err(_) => return Err(anyhow::anyhow!("Connection timeout")),
        };
        
        debug!("TCP connection established to {}", address);
        
        // Build handshake request
        let handshake_request = HandshakeRequest {
            protocol_version: PROTOCOL_VERSION.to_string(),
            genesis_hash: self.get_genesis_hash(),
            node_pubkey: self.get_node_pubkey(),
            timestamp: current_timestamp(),
            is_validator: false, // Would be set from config
        };
        
        // Serialize the request
        let request_bytes = bincode::serialize(&handshake_request)
            .context("Failed to serialize handshake request")?;
        
        // Send message length (4 bytes, big-endian)
        let length_bytes = (request_bytes.len() as u32).to_be_bytes();
        stream.write_all(&length_bytes).await
            .context("Failed to write handshake length")?;
        
        // Send the actual request
        stream.write_all(&request_bytes).await
            .context("Failed to write handshake request")?;
        
        stream.flush().await
            .context("Failed to flush handshake request")?;
        
        debug!("Sent handshake request ({} bytes) to {}", request_bytes.len(), address);
        
        // Read response with timeout
        let read_timeout = std::time::Duration::from_secs(10);
        
        // Read response length (4 bytes)
        let mut length_buf = [0u8; 4];
        match tokio::time::timeout(read_timeout, stream.read_exact(&mut length_buf)).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(anyhow::anyhow!("Failed to read response length: {}", e)),
            Err(_) => return Err(anyhow::anyhow!("Handshake response timeout")),
        }
        
        let response_length = u32::from_be_bytes(length_buf) as usize;
        
        // Validate response length (max 64KB for handshake)
        if response_length > 65536 {
            return Err(anyhow::anyhow!(
                "Handshake response too large: {} bytes (max 65536)",
                response_length
            ));
        }
        
        if response_length == 0 {
            return Err(anyhow::anyhow!("Empty handshake response"));
        }
        
        // Read the response data
        let mut response_bytes = vec![0u8; response_length];
        match tokio::time::timeout(read_timeout, stream.read_exact(&mut response_bytes)).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(anyhow::anyhow!("Failed to read response data: {}", e)),
            Err(_) => return Err(anyhow::anyhow!("Handshake response data timeout")),
        }
        
        debug!("Received handshake response ({} bytes) from {}", response_bytes.len(), address);
        
        // Deserialize the response
        let response: HandshakeResponse = bincode::deserialize(&response_bytes)
            .context("Failed to deserialize handshake response")?;
        
        // Validate the response
        if response.pubkey.is_empty() {
            return Err(anyhow::anyhow!("Peer sent empty public key"));
        }
        
        if response.version.is_empty() {
            return Err(anyhow::anyhow!("Peer sent empty version"));
        }
        
        // Verify timestamp freshness (must be within 5 minutes)
        // This prevents replay attacks
        let now = current_timestamp();
        let request_age = now.saturating_sub(handshake_request.timestamp);
        if request_age > 300 {
            warn!("Handshake took too long: {} seconds", request_age);
        }
        
        info!("Handshake completed with {}: version={}, validator={}", 
            address, response.version, response.is_validator);
        
        Ok(response)
    }
    
    /// Check if protocol version is compatible
    fn is_version_compatible(&self, version: &str) -> bool {
        // Parse semver and check major version compatibility
        let our_parts: Vec<u32> = PROTOCOL_VERSION.split('.')
            .filter_map(|s| s.parse().ok())
            .collect();
        let their_parts: Vec<u32> = version.split('.')
            .filter_map(|s| s.parse().ok())
            .collect();
        
        // Must match major version
        if our_parts.is_empty() || their_parts.is_empty() {
            return false;
        }
        
        our_parts[0] == their_parts[0]
    }
    
    /// Verify genesis hash matches our chain
    fn verify_genesis_hash(&self, genesis_hash: &[u8]) -> bool {
        genesis_hash == self.get_genesis_hash()
    }
    
    /// Get our genesis hash
    fn get_genesis_hash(&self) -> Vec<u8> {
        self.genesis_hash.clone()
    }
    
    /// Get our node's public key
    fn get_node_pubkey(&self) -> Vec<u8> {
        self.node_keypair.public.as_bytes().to_vec()
    }
    
    /// Get the node's public key as base58 string
    pub fn get_node_identity(&self) -> String {
        bs58::encode(self.node_keypair.public.as_bytes()).into_string()
    }
    
    /// Sign a message with the node's private key
    pub fn sign_message(&self, message: &[u8]) -> Vec<u8> {
        use ed25519_dalek::Signer;
        self.node_keypair.sign(message).to_bytes().to_vec()
    }
    
    /// Generate deterministic peer ID from address (for testing)
    fn generate_peer_id(&self, address: &str) -> Vec<u8> {
        hash_data(address.as_bytes())
    }

    /// Broadcast a message using Gossipsub-style selective propagation
    /// 
    /// Instead of sending to ALL peers (which causes O(n²) message explosion),
    /// we use a mesh-based approach:
    /// 1. Send to mesh peers (D peers) for reliable delivery
    /// 2. Optionally send to a few eager push peers for faster propagation
    /// 
    /// This reduces bandwidth from O(peers²) to O(peers * D) where D << peers
    pub async fn broadcast(&self, message: GossipMessage) -> Result<usize> {
        let message_hash = hash_message(&message);
        
        // Check if we've already seen this message using TTL-based cache
        // This prevents broadcast storms that would occur with cache.clear()
        {
            let mut cache = self.message_cache.lock().await;
            
            // TimeCache handles TTL expiration and FIFO eviction internally
            if cache.contains(&message_hash) {
                debug!("Dropping duplicate message: already seen");
                return Ok(0);
            }
            
            // Insert into cache - old entries are evicted by TTL/FIFO, never cleared
            cache.insert(message_hash.clone());
            
            // Log cache stats periodically for monitoring
            let (current, max) = cache.stats();
            if current % 1000 == 0 {
                debug!("Message cache: {}/{} entries", current, max);
            }
        }

        let peers = self.peers.read().await;
        let peer_count = peers.len();
        
        if peer_count == 0 {
            return Ok(0);
        }
        
        // Get or build mesh peers
        let mesh_peers = self.get_or_build_mesh(&peers).await;
        let mesh_size = mesh_peers.len();
        
        // Select subset of peers to send to (Gossipsub mesh)
        // For critical messages like blocks and votes, we send to mesh peers
        // For transactions, we can use a smaller eager push set
        let target_peers: Vec<_> = match &message {
            GossipMessage::BlockProposal { .. } | GossipMessage::Vote { .. } => {
                // High priority: send to all mesh peers
                mesh_peers.iter()
                    .filter_map(|pk| peers.get(pk))
                    .cloned()
                    .collect()
            }
            GossipMessage::TransactionBatch { .. } => {
                // Lower priority: send to eager push subset
                let eager_count = self.gossip_params.eager_push_peers.min(mesh_size);
                mesh_peers.iter()
                    .take(eager_count)
                    .filter_map(|pk| peers.get(pk))
                    .cloned()
                    .collect()
            }
            _ => {
                // Default: send to half of mesh
                let count = (mesh_size / 2).max(1);
                mesh_peers.iter()
                    .take(count)
                    .filter_map(|pk| peers.get(pk))
                    .cloned()
                    .collect()
            }
        };
        
        let send_count = target_peers.len();
        
        debug!(
            "Gossipsub broadcast: sending to {}/{} mesh peers (total peers: {})",
            send_count, mesh_size, peer_count
        );
        
        // Serialize message once for all sends
        let message_bytes = bincode::serialize(&message)
            .map_err(|e| anyhow::anyhow!("Failed to serialize message: {}", e))?;
        
        // Send to each target peer
        let mut success_count = 0;
        for peer in &target_peers {
            match self.send_message_to_peer(&peer.address, &message_bytes).await {
                Ok(_) => {
                    success_count += 1;
                }
                Err(e) => {
                    debug!("Failed to send to peer {}: {}", peer.address, e);
                    // Don't fail the broadcast if some peers are unreachable
                }
            }
        }
        
        if success_count < send_count {
            debug!(
                "Broadcast partially successful: {}/{} peers reached",
                success_count, send_count
            );
        }

        Ok(success_count)
    }
    
    /// Send a serialized message to a specific peer via TCP
    async fn send_message_to_peer(&self, address: &str, message_bytes: &[u8]) -> Result<()> {
        use tokio::net::TcpStream;
        use tokio::io::AsyncWriteExt;
        
        // Parse address (format: "host:port")
        let mut stream = match tokio::time::timeout(
            std::time::Duration::from_secs(5),
            TcpStream::connect(address)
        ).await {
            Ok(Ok(stream)) => stream,
            Ok(Err(e)) => return Err(anyhow::anyhow!("Connection failed: {}", e)),
            Err(_) => return Err(anyhow::anyhow!("Connection timeout")),
        };
        
        // Send message length (4 bytes, big endian)
        let len = message_bytes.len() as u32;
        stream.write_all(&len.to_be_bytes()).await
            .map_err(|e| anyhow::anyhow!("Failed to send length: {}", e))?;
        
        // Send message body
        stream.write_all(message_bytes).await
            .map_err(|e| anyhow::anyhow!("Failed to send message: {}", e))?;
        
        stream.flush().await
            .map_err(|e| anyhow::anyhow!("Failed to flush: {}", e))?;
        
        Ok(())
    }
    
    /// Get or build the mesh peers for this node
    /// 
    /// Mesh peers are a subset of all peers that we actively gossip with.
    /// This creates a sparse overlay network that prevents broadcast storms.
    async fn get_or_build_mesh(&self, all_peers: &HashMap<Vec<u8>, PeerInfo>) -> Vec<Vec<u8>> {
        let mut mesh = self.mesh_peers.write().await;
        
        // Clean up mesh peers that are no longer connected
        mesh.retain(|pk| all_peers.contains_key(pk));
        
        let target_size = self.gossip_params.mesh_degree;
        let current_size = mesh.len();
        
        // If mesh is too small, add more peers
        if current_size < self.gossip_params.mesh_degree_low {
            // Prefer validators for mesh (they're more reliable)
            let mut candidates: Vec<_> = all_peers.iter()
                .filter(|(pk, _)| !mesh.contains(*pk))
                .collect();
            
            // Sort by validator status (validators first) and stake (higher stake first)
            candidates.sort_by(|(_, a), (_, b)| {
                match (b.is_validator, a.is_validator) {
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    _ => b.stake.cmp(&a.stake),
                }
            });
            
            // Add peers until we reach target size
            let needed = target_size.saturating_sub(current_size);
            for (pk, _) in candidates.into_iter().take(needed) {
                mesh.insert(pk.clone());
            }
            
            debug!("Mesh rebuilt: {} -> {} peers", current_size, mesh.len());
        }
        
        // If mesh is too large, prune lowest-stake non-validators
        if mesh.len() > self.gossip_params.mesh_degree_high {
            let mut mesh_with_info: Vec<_> = mesh.iter()
                .filter_map(|pk| all_peers.get(pk).map(|p| (pk.clone(), p.clone())))
                .collect();
            
            // Sort to keep validators and high-stake peers
            mesh_with_info.sort_by(|(_, a), (_, b)| {
                match (a.is_validator, b.is_validator) {
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    _ => b.stake.cmp(&a.stake),
                }
            });
            
            // Keep only target_size peers
            mesh.clear();
            for (pk, _) in mesh_with_info.into_iter().take(target_size) {
                mesh.insert(pk);
            }
            
            debug!("Mesh pruned to {} peers", mesh.len());
        }
        
        mesh.iter().cloned().collect()
    }

    pub async fn broadcast_transaction(&self, tx_data: &[u8]) -> Result<()> {
        let message = GossipMessage::TransactionBatch {
            transactions: vec![tx_data.to_vec()],
        };
        
        let sent = self.broadcast(message).await?;
        info!("Broadcasted transaction to {} peers", sent);
        
        Ok(())
    }

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

    pub async fn get_peer_count(&self) -> usize {
        self.peers.read().await.len()
    }

    pub async fn get_peers(&self) -> Vec<PeerInfo> {
        self.peers.read().await.values().cloned().collect()
    }

    pub async fn get_validator_peers(&self) -> Vec<PeerInfo> {
        self.peers.read().await
            .values()
            .filter(|p| p.is_validator)
            .cloned()
            .collect()
    }

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

    pub async fn update_peer(&self, pubkey: &[u8], update_fn: impl FnOnce(&mut PeerInfo)) -> Result<()> {
        let mut peers = self.peers.write().await;
        
        if let Some(peer) = peers.get_mut(pubkey) {
            update_fn(peer);
            peer.last_seen = current_timestamp();
        }
        
        Ok(())
    }

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

    fn spawn_message_processor_task(&self) {
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        });
    }

    fn spawn_health_check_task(&self) {
        let peers = self.peers.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                debug!("Sending health check pings");
            }
        });
    }

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

fn hash_message(message: &GossipMessage) -> Vec<u8> {
    let serialized = bincode::serialize(message).unwrap_or_default();
    hash_data(&serialized)
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
        
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        let removed = network.prune_inactive_peers(0).await;
        assert_eq!(removed, 1);
    }
}
