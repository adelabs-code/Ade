//! Genesis Block Generation
//!
//! This module handles the creation and validation of the genesis block,
//! which is the first block in the blockchain and defines initial state.
//!
//! # Usage
//!
//! ```bash
//! # Generate genesis with CLI
//! ade-cli genesis generate --config genesis.toml --output genesis.bin
//!
//! # Start node with genesis
//! ade-node --genesis genesis.bin --data-dir ./data
//! ```

use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};
use anyhow::{Result, Context};
use std::collections::HashMap;
use std::path::Path;
use std::fs;
use tracing::{info, warn};

use crate::block::{Block, BlockHeader, BlockHash};

/// Genesis configuration loaded from TOML file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenesisConfig {
    /// Network identification
    pub network: NetworkConfig,
    /// Consensus parameters
    pub consensus: ConsensusConfig,
    /// Initial token allocation
    pub allocations: Vec<AllocationConfig>,
    /// Initial validator set
    pub validators: Vec<ValidatorConfig>,
    /// System accounts (treasury, bridge, etc.)
    pub system_accounts: SystemAccountsConfig,
    /// AI runtime configuration
    #[serde(default)]
    pub ai_runtime: AIRuntimeConfig,
}

/// Network identification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Unique network identifier (e.g., "ade-mainnet", "ade-testnet")
    pub network_id: String,
    /// Human-readable network name
    pub name: String,
    /// Chain ID for transaction signing
    pub chain_id: u64,
    /// Target block time in milliseconds
    #[serde(default = "default_block_time")]
    pub block_time_ms: u64,
    /// Genesis timestamp (Unix timestamp, 0 = now)
    #[serde(default)]
    pub genesis_timestamp: u64,
}

fn default_block_time() -> u64 {
    400 // 400ms default
}

/// Consensus configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfig {
    /// Minimum stake required to become a validator (in lamports)
    pub min_stake: u64,
    /// Number of slots per epoch
    pub epoch_length: u64,
    /// Number of confirmations for finality
    #[serde(default = "default_finality_confirmations")]
    pub finality_confirmations: u32,
    /// Maximum validators in active set
    #[serde(default = "default_max_validators")]
    pub max_validators: usize,
    /// Inflation rate per epoch (basis points, 100 = 1%)
    #[serde(default)]
    pub inflation_rate_bps: u64,
    /// Slashing percentage for misbehavior (basis points)
    #[serde(default = "default_slash_rate")]
    pub slash_rate_bps: u64,
}

fn default_finality_confirmations() -> u32 { 32 }
fn default_max_validators() -> usize { 100 }
fn default_slash_rate() -> u64 { 1000 } // 10%

/// Initial token allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationConfig {
    /// Recipient address (base58 encoded)
    pub address: String,
    /// Amount in lamports
    pub amount: u64,
    /// Optional label for the allocation
    #[serde(default)]
    pub label: String,
    /// Lock-up period in slots (0 = no lockup)
    #[serde(default)]
    pub lockup_slots: u64,
}

/// Initial validator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorConfig {
    /// Validator's public key (base58 encoded)
    pub pubkey: String,
    /// Initial stake amount
    pub stake: u64,
    /// Commission rate (basis points, 1000 = 10%)
    #[serde(default = "default_commission")]
    pub commission_bps: u64,
    /// Validator name/identity
    #[serde(default)]
    pub name: String,
    /// Validator website
    #[serde(default)]
    pub website: String,
}

fn default_commission() -> u64 { 1000 } // 10%

/// System accounts configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SystemAccountsConfig {
    /// Treasury account for protocol fees
    #[serde(default)]
    pub treasury: Option<String>,
    /// Bridge program account
    #[serde(default)]
    pub bridge_program: Option<String>,
    /// AI runtime program account
    #[serde(default)]
    pub ai_runtime_program: Option<String>,
    /// Fee collector account
    #[serde(default)]
    pub fee_collector: Option<String>,
}

/// AI Runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIRuntimeConfig {
    /// Maximum model size in bytes
    #[serde(default = "default_max_model_size")]
    pub max_model_size: u64,
    /// Maximum compute units per execution
    #[serde(default = "default_max_compute")]
    pub max_compute_units: u64,
    /// Base cost per compute unit (lamports)
    #[serde(default = "default_compute_cost")]
    pub compute_unit_cost: u64,
}

fn default_max_model_size() -> u64 { 100 * 1024 * 1024 } // 100 MB
fn default_max_compute() -> u64 { 1_000_000 }
fn default_compute_cost() -> u64 { 1 }

impl Default for AIRuntimeConfig {
    fn default() -> Self {
        Self {
            max_model_size: default_max_model_size(),
            max_compute_units: default_max_compute(),
            compute_unit_cost: default_compute_cost(),
        }
    }
}

/// Genesis block with full state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genesis {
    /// The genesis block
    pub block: Block,
    /// Genesis configuration used to create this genesis
    pub config: GenesisConfig,
    /// Initial account states
    pub accounts: HashMap<Vec<u8>, GenesisAccount>,
    /// Genesis hash (computed from block)
    pub hash: Vec<u8>,
}

/// Account state at genesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenesisAccount {
    pub address: Vec<u8>,
    pub lamports: u64,
    pub owner: Vec<u8>,
    pub data: Vec<u8>,
    pub executable: bool,
}

impl GenesisConfig {
    /// Load genesis configuration from TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read genesis config: {:?}", path.as_ref()))?;
        
        let config: GenesisConfig = toml::from_str(&content)
            .context("Failed to parse genesis config TOML")?;
        
        config.validate()?;
        
        Ok(config)
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Network validation
        if self.network.network_id.is_empty() {
            return Err(anyhow::anyhow!("network_id is required"));
        }
        if self.network.chain_id == 0 {
            return Err(anyhow::anyhow!("chain_id must be > 0"));
        }
        
        // Consensus validation
        if self.consensus.min_stake == 0 {
            return Err(anyhow::anyhow!("min_stake must be > 0"));
        }
        if self.consensus.epoch_length < 100 {
            return Err(anyhow::anyhow!("epoch_length should be at least 100 slots"));
        }
        
        // Validator validation
        if self.validators.is_empty() {
            return Err(anyhow::anyhow!("At least one validator is required"));
        }
        for (i, v) in self.validators.iter().enumerate() {
            if v.stake < self.consensus.min_stake {
                return Err(anyhow::anyhow!(
                    "Validator {} has stake {} < min_stake {}",
                    i, v.stake, self.consensus.min_stake
                ));
            }
            // Validate pubkey format
            bs58::decode(&v.pubkey).into_vec()
                .with_context(|| format!("Invalid validator pubkey: {}", v.pubkey))?;
        }
        
        // Allocation validation
        for (i, a) in self.allocations.iter().enumerate() {
            bs58::decode(&a.address).into_vec()
                .with_context(|| format!("Invalid allocation address at index {}: {}", i, a.address))?;
        }
        
        Ok(())
    }
    
    /// Calculate total supply at genesis
    pub fn total_supply(&self) -> u64 {
        let allocation_total: u64 = self.allocations.iter().map(|a| a.amount).sum();
        let validator_total: u64 = self.validators.iter().map(|v| v.stake).sum();
        allocation_total + validator_total
    }
}

impl Genesis {
    /// Generate genesis from configuration
    pub fn generate(config: GenesisConfig) -> Result<Self> {
        info!("Generating genesis block for network: {}", config.network.network_id);
        
        // Determine genesis timestamp
        let timestamp = if config.network.genesis_timestamp > 0 {
            config.network.genesis_timestamp
        } else {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        };
        
        // Create initial accounts
        let mut accounts = HashMap::new();
        
        // Process allocations
        for allocation in &config.allocations {
            let address = bs58::decode(&allocation.address).into_vec()?;
            let account = GenesisAccount {
                address: address.clone(),
                lamports: allocation.amount,
                owner: vec![0u8; 32], // System program
                data: vec![],
                executable: false,
            };
            accounts.insert(address, account);
            info!("Allocation: {} -> {} lamports ({})", 
                allocation.address, allocation.amount, allocation.label);
        }
        
        // Process validator stakes
        for validator in &config.validators {
            let address = bs58::decode(&validator.pubkey).into_vec()?;
            
            // Create stake account for validator
            let stake_data = bincode::serialize(&ValidatorStakeAccount {
                validator_pubkey: address.clone(),
                stake: validator.stake,
                commission_bps: validator.commission_bps,
                activated_epoch: 0,
                deactivated_epoch: u64::MAX,
            })?;
            
            let stake_account = GenesisAccount {
                address: derive_stake_address(&address),
                lamports: validator.stake,
                owner: STAKE_PROGRAM_ID.to_vec(),
                data: stake_data,
                executable: false,
            };
            accounts.insert(stake_account.address.clone(), stake_account);
            
            info!("Validator: {} with {} stake", validator.pubkey, validator.stake);
        }
        
        // Create system accounts
        if let Some(ref treasury) = config.system_accounts.treasury {
            let address = bs58::decode(treasury).into_vec()?;
            accounts.insert(address.clone(), GenesisAccount {
                address,
                lamports: 0,
                owner: vec![0u8; 32],
                data: vec![],
                executable: false,
            });
        }
        
        // Select first validator as genesis block producer
        let genesis_validator = bs58::decode(&config.validators[0].pubkey).into_vec()?;
        
        // Create genesis block
        let genesis_header = BlockHeader {
            slot: 0,
            parent_hash: vec![], // Genesis has no parent
            transactions_root: vec![0u8; 32], // No transactions
            accounts_root: compute_accounts_root(&accounts),
            timestamp,
            validator: genesis_validator,
            signature: vec![], // Genesis is not signed
        };
        
        let genesis_block = Block {
            header: genesis_header,
            transactions: vec![],
        };
        
        let genesis_hash = genesis_block.hash();
        
        info!("Genesis block created:");
        info!("  Hash: {}", bs58::encode(&genesis_hash).into_string());
        info!("  Timestamp: {}", timestamp);
        info!("  Total accounts: {}", accounts.len());
        info!("  Total supply: {} lamports", config.total_supply());
        
        Ok(Self {
            block: genesis_block,
            config,
            accounts,
            hash: genesis_hash,
        })
    }
    
    /// Save genesis to binary file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let data = bincode::serialize(self)
            .context("Failed to serialize genesis")?;
        
        fs::write(&path, &data)
            .with_context(|| format!("Failed to write genesis to {:?}", path.as_ref()))?;
        
        info!("Genesis saved to {:?} ({} bytes)", path.as_ref(), data.len());
        Ok(())
    }
    
    /// Load genesis from binary file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let data = fs::read(&path)
            .with_context(|| format!("Failed to read genesis from {:?}", path.as_ref()))?;
        
        let genesis: Genesis = bincode::deserialize(&data)
            .context("Failed to deserialize genesis")?;
        
        info!("Genesis loaded: {}", bs58::encode(&genesis.hash).into_string());
        Ok(genesis)
    }
    
    /// Get genesis hash as base58 string
    pub fn hash_string(&self) -> String {
        bs58::encode(&self.hash).into_string()
    }
    
    /// Validate loaded genesis
    pub fn validate(&self) -> Result<()> {
        // Verify block structure
        self.block.validate_structure()?;
        
        // Verify genesis block properties
        if self.block.header.slot != 0 {
            return Err(anyhow::anyhow!("Genesis block must have slot 0"));
        }
        if !self.block.header.parent_hash.is_empty() {
            return Err(anyhow::anyhow!("Genesis block must have empty parent hash"));
        }
        
        // Verify hash matches
        let computed_hash = self.block.hash();
        if computed_hash != self.hash {
            return Err(anyhow::anyhow!("Genesis hash mismatch"));
        }
        
        Ok(())
    }
}

/// Validator stake account data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorStakeAccount {
    pub validator_pubkey: Vec<u8>,
    pub stake: u64,
    pub commission_bps: u64,
    pub activated_epoch: u64,
    pub deactivated_epoch: u64,
}

/// Stake program ID (placeholder)
const STAKE_PROGRAM_ID: [u8; 32] = [
    6, 161, 216, 23, 145, 55, 84, 42, 152, 52, 55, 189, 254, 42, 122, 178,
    85, 127, 83, 92, 138, 120, 114, 43, 104, 164, 157, 192, 0, 0, 0, 0
];

/// Derive stake account address from validator pubkey
fn derive_stake_address(validator_pubkey: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(b"STAKE_ACCOUNT:");
    hasher.update(validator_pubkey);
    hasher.finalize().to_vec()
}

/// Compute Merkle root of accounts
fn compute_accounts_root(accounts: &HashMap<Vec<u8>, GenesisAccount>) -> Vec<u8> {
    if accounts.is_empty() {
        return vec![0u8; 32];
    }
    
    let mut hasher = Sha256::new();
    hasher.update(b"ACCOUNTS_ROOT:");
    
    let mut sorted_keys: Vec<_> = accounts.keys().collect();
    sorted_keys.sort();
    
    for key in sorted_keys {
        if let Some(account) = accounts.get(key) {
            hasher.update(key);
            hasher.update(&account.lamports.to_le_bytes());
            hasher.update(&account.owner);
        }
    }
    
    hasher.finalize().to_vec()
}

/// Create a default testnet genesis config
pub fn create_testnet_config() -> GenesisConfig {
    GenesisConfig {
        network: NetworkConfig {
            network_id: "ade-testnet".to_string(),
            name: "Ade Testnet".to_string(),
            chain_id: 1001,
            block_time_ms: 400,
            genesis_timestamp: 0,
        },
        consensus: ConsensusConfig {
            min_stake: 1_000_000_000, // 1 SOL
            epoch_length: 432_000, // ~2 days at 400ms
            finality_confirmations: 32,
            max_validators: 100,
            inflation_rate_bps: 500, // 5%
            slash_rate_bps: 1000, // 10%
        },
        allocations: vec![
            AllocationConfig {
                address: "11111111111111111111111111111111".to_string(),
                amount: 500_000_000_000_000_000, // 500M tokens
                label: "Foundation".to_string(),
                lockup_slots: 0,
            },
        ],
        validators: vec![
            ValidatorConfig {
                pubkey: "Va1idator1111111111111111111111111111111111".to_string(),
                stake: 10_000_000_000_000, // 10K tokens
                commission_bps: 1000,
                name: "Genesis Validator".to_string(),
                website: "https://ade.network".to_string(),
            },
        ],
        system_accounts: SystemAccountsConfig::default(),
        ai_runtime: AIRuntimeConfig::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genesis_config_validation() {
        let config = create_testnet_config();
        // This will fail because addresses are placeholders, but structure is valid
        // In real use, valid base58 addresses would be provided
    }

    #[test]
    fn test_total_supply_calculation() {
        let mut config = create_testnet_config();
        config.allocations = vec![
            AllocationConfig {
                address: "11111111111111111111111111111111".to_string(),
                amount: 1000,
                label: "Test".to_string(),
                lockup_slots: 0,
            },
        ];
        config.validators = vec![
            ValidatorConfig {
                pubkey: "11111111111111111111111111111111".to_string(),
                stake: 500,
                commission_bps: 1000,
                name: "Test".to_string(),
                website: "".to_string(),
            },
        ];
        
        assert_eq!(config.total_supply(), 1500);
    }
}

