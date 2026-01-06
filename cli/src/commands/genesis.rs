//! Genesis CLI Commands
//!
//! Commands for generating and managing genesis blocks.

use clap::{Args, Subcommand};
use anyhow::{Result, Context};
use std::path::PathBuf;
use tracing::{info, warn};

#[derive(Debug, Args)]
pub struct GenesisArgs {
    #[command(subcommand)]
    pub command: GenesisCommand,
}

#[derive(Debug, Subcommand)]
pub enum GenesisCommand {
    /// Generate a new genesis block from configuration
    Generate {
        /// Path to genesis configuration file (TOML)
        #[arg(short, long, default_value = "genesis.toml")]
        config: PathBuf,
        
        /// Output path for genesis binary file
        #[arg(short, long, default_value = "genesis.bin")]
        output: PathBuf,
        
        /// Also output genesis hash to file
        #[arg(long)]
        hash_file: Option<PathBuf>,
    },
    
    /// Create a sample genesis configuration file
    Init {
        /// Output path for sample config
        #[arg(short, long, default_value = "genesis.toml")]
        output: PathBuf,
        
        /// Network type: mainnet, testnet, or devnet
        #[arg(short, long, default_value = "testnet")]
        network: String,
        
        /// Your validator public key (base58)
        #[arg(long)]
        validator_pubkey: Option<String>,
        
        /// Initial validator stake
        #[arg(long, default_value = "10000000000000")]
        validator_stake: u64,
    },
    
    /// Verify a genesis binary file
    Verify {
        /// Path to genesis binary file
        #[arg(short, long, default_value = "genesis.bin")]
        genesis: PathBuf,
    },
    
    /// Show genesis information
    Info {
        /// Path to genesis binary file
        #[arg(short, long, default_value = "genesis.bin")]
        genesis: PathBuf,
        
        /// Show full account list
        #[arg(long)]
        accounts: bool,
    },
    
    /// Export genesis hash
    Hash {
        /// Path to genesis binary file
        #[arg(short, long, default_value = "genesis.bin")]
        genesis: PathBuf,
        
        /// Output format: base58, hex, or json
        #[arg(short, long, default_value = "base58")]
        format: String,
    },
}

pub async fn execute(args: GenesisArgs) -> Result<()> {
    match args.command {
        GenesisCommand::Generate { config, output, hash_file } => {
            generate_genesis(&config, &output, hash_file.as_deref()).await
        }
        GenesisCommand::Init { output, network, validator_pubkey, validator_stake } => {
            init_genesis_config(&output, &network, validator_pubkey, validator_stake).await
        }
        GenesisCommand::Verify { genesis } => {
            verify_genesis(&genesis).await
        }
        GenesisCommand::Info { genesis, accounts } => {
            show_genesis_info(&genesis, accounts).await
        }
        GenesisCommand::Hash { genesis, format } => {
            show_genesis_hash(&genesis, &format).await
        }
    }
}

async fn generate_genesis(
    config_path: &PathBuf,
    output_path: &PathBuf,
    hash_file: Option<&PathBuf>,
) -> Result<()> {
    use ade_consensus::genesis::{GenesisConfig, Genesis};
    
    info!("Loading genesis configuration from {:?}", config_path);
    
    let config = GenesisConfig::from_file(config_path)?;
    
    info!("Generating genesis block...");
    let genesis = Genesis::generate(config)?;
    
    info!("Saving genesis to {:?}", output_path);
    genesis.save(output_path)?;
    
    // Optionally save hash to file
    if let Some(hash_path) = hash_file {
        let hash_str = genesis.hash_string();
        std::fs::write(hash_path, &hash_str)
            .with_context(|| format!("Failed to write hash to {:?}", hash_path))?;
        info!("Genesis hash saved to {:?}", hash_path);
    }
    
    println!("\n✅ Genesis block generated successfully!");
    println!("   Hash: {}", genesis.hash_string());
    println!("   Output: {:?}", output_path);
    println!("   Accounts: {}", genesis.accounts.len());
    println!("   Total supply: {} lamports", genesis.config.total_supply());
    
    Ok(())
}

async fn init_genesis_config(
    output_path: &PathBuf,
    network: &str,
    validator_pubkey: Option<String>,
    validator_stake: u64,
) -> Result<()> {
    use ade_consensus::genesis::*;
    
    info!("Creating sample genesis configuration for {}", network);
    
    let (network_id, chain_id, name) = match network {
        "mainnet" => ("ade-mainnet", 1, "Ade Mainnet"),
        "testnet" => ("ade-testnet", 1001, "Ade Testnet"),
        "devnet" => ("ade-devnet", 1002, "Ade Devnet"),
        _ => {
            warn!("Unknown network type '{}', using custom", network);
            (network, 9999, "Ade Custom")
        }
    };
    
    // Generate a placeholder keypair if not provided
    let validator_pubkey = validator_pubkey.unwrap_or_else(|| {
        warn!("No validator pubkey provided, using placeholder");
        warn!("Replace this with your actual validator pubkey before generating genesis!");
        "REPLACE_WITH_YOUR_VALIDATOR_PUBKEY_BASE58".to_string()
    });
    
    let config = GenesisConfig {
        network: NetworkConfig {
            network_id: network_id.to_string(),
            name: name.to_string(),
            chain_id,
            block_time_ms: 400,
            genesis_timestamp: 0, // Will use current time
        },
        consensus: ConsensusConfig {
            min_stake: 1_000_000_000, // 1 SOL
            epoch_length: if network == "mainnet" { 432_000 } else { 8640 }, // 2 days / 1 hour
            finality_confirmations: 32,
            max_validators: if network == "mainnet" { 200 } else { 50 },
            inflation_rate_bps: if network == "mainnet" { 300 } else { 500 }, // 3% / 5%
            slash_rate_bps: 1000, // 10%
        },
        allocations: vec![
            AllocationConfig {
                address: "REPLACE_WITH_TREASURY_ADDRESS".to_string(),
                amount: if network == "mainnet" {
                    500_000_000_000_000_000 // 500M tokens
                } else {
                    1_000_000_000_000_000 // 1M tokens for testing
                },
                label: "Treasury".to_string(),
                lockup_slots: 0,
            },
            AllocationConfig {
                address: "REPLACE_WITH_TEAM_ADDRESS".to_string(),
                amount: if network == "mainnet" {
                    100_000_000_000_000_000 // 100M tokens
                } else {
                    100_000_000_000_000 // 100K tokens for testing
                },
                label: "Team".to_string(),
                lockup_slots: if network == "mainnet" { 
                    432_000 * 365 // 1 year lockup 
                } else { 
                    0 
                },
            },
        ],
        validators: vec![
            ValidatorConfig {
                pubkey: validator_pubkey,
                stake: validator_stake,
                commission_bps: 1000, // 10%
                name: "Genesis Validator".to_string(),
                website: "https://ade.network".to_string(),
            },
        ],
        system_accounts: SystemAccountsConfig {
            treasury: Some("REPLACE_WITH_TREASURY_ADDRESS".to_string()),
            bridge_program: Some("REPLACE_WITH_BRIDGE_PROGRAM_ID".to_string()),
            ai_runtime_program: Some("REPLACE_WITH_AI_RUNTIME_PROGRAM_ID".to_string()),
            fee_collector: Some("REPLACE_WITH_FEE_COLLECTOR_ADDRESS".to_string()),
        },
        ai_runtime: AIRuntimeConfig {
            max_model_size: 100 * 1024 * 1024, // 100 MB
            max_compute_units: 1_000_000,
            compute_unit_cost: 1,
        },
    };
    
    // Serialize to TOML
    let toml_string = toml::to_string_pretty(&config)
        .context("Failed to serialize config to TOML")?;
    
    // Add helpful comments
    let config_with_comments = format!(
r#"# Ade Genesis Configuration
# Network: {name}
# 
# IMPORTANT: Replace all placeholder addresses before generating genesis!
# 
# Generate genesis block:
#   ade-cli genesis generate --config genesis.toml --output genesis.bin
#
# Start node with genesis:
#   ade-node --genesis genesis.bin --data-dir ./data

{}"#,
        toml_string,
        name = name
    );
    
    std::fs::write(output_path, config_with_comments)
        .with_context(|| format!("Failed to write config to {:?}", output_path))?;
    
    println!("\n✅ Genesis configuration created: {:?}", output_path);
    println!("\n⚠️  IMPORTANT: Edit the configuration file to:");
    println!("   1. Replace placeholder addresses with real ones");
    println!("   2. Set your validator pubkey");
    println!("   3. Configure token allocations");
    println!("   4. Adjust consensus parameters if needed");
    println!("\nThen run: ade-cli genesis generate --config {:?}", output_path);
    
    Ok(())
}

async fn verify_genesis(genesis_path: &PathBuf) -> Result<()> {
    use ade_consensus::genesis::Genesis;
    
    info!("Loading genesis from {:?}", genesis_path);
    let genesis = Genesis::load(genesis_path)?;
    
    info!("Validating genesis...");
    genesis.validate()?;
    
    println!("\n✅ Genesis is valid!");
    println!("   Hash: {}", genesis.hash_string());
    println!("   Network: {}", genesis.config.network.network_id);
    println!("   Chain ID: {}", genesis.config.network.chain_id);
    
    Ok(())
}

async fn show_genesis_info(genesis_path: &PathBuf, show_accounts: bool) -> Result<()> {
    use ade_consensus::genesis::Genesis;
    
    let genesis = Genesis::load(genesis_path)?;
    
    println!("\n📦 Genesis Block Information");
    println!("============================");
    println!();
    println!("Network:");
    println!("  ID:        {}", genesis.config.network.network_id);
    println!("  Name:      {}", genesis.config.network.name);
    println!("  Chain ID:  {}", genesis.config.network.chain_id);
    println!("  Block Time: {}ms", genesis.config.network.block_time_ms);
    println!();
    println!("Block:");
    println!("  Hash:      {}", genesis.hash_string());
    println!("  Slot:      {}", genesis.block.header.slot);
    println!("  Timestamp: {}", genesis.block.header.timestamp);
    println!("  Validator: {}", bs58::encode(&genesis.block.header.validator).into_string());
    println!();
    println!("Consensus:");
    println!("  Min Stake:    {} lamports", genesis.config.consensus.min_stake);
    println!("  Epoch Length: {} slots", genesis.config.consensus.epoch_length);
    println!("  Finality:     {} confirmations", genesis.config.consensus.finality_confirmations);
    println!("  Max Validators: {}", genesis.config.consensus.max_validators);
    println!();
    println!("Token Supply:");
    println!("  Total: {} lamports", genesis.config.total_supply());
    println!("  Allocations: {}", genesis.config.allocations.len());
    println!();
    println!("Validators ({}):", genesis.config.validators.len());
    for (i, v) in genesis.config.validators.iter().enumerate() {
        println!("  {}. {} (stake: {}, commission: {}%)", 
            i + 1, 
            if v.name.is_empty() { &v.pubkey } else { &v.name },
            v.stake,
            v.commission_bps as f64 / 100.0
        );
    }
    
    if show_accounts {
        println!();
        println!("Accounts ({}):", genesis.accounts.len());
        for (address, account) in &genesis.accounts {
            println!("  {} : {} lamports",
                bs58::encode(address).into_string(),
                account.lamports
            );
        }
    }
    
    Ok(())
}

async fn show_genesis_hash(genesis_path: &PathBuf, format: &str) -> Result<()> {
    use ade_consensus::genesis::Genesis;
    
    let genesis = Genesis::load(genesis_path)?;
    
    match format {
        "base58" => println!("{}", genesis.hash_string()),
        "hex" => println!("{}", hex::encode(&genesis.hash)),
        "json" => println!(r#"{{"hash":"{}","format":"base58"}}"#, genesis.hash_string()),
        _ => {
            return Err(anyhow::anyhow!("Unknown format: {}. Use base58, hex, or json", format));
        }
    }
    
    Ok(())
}

