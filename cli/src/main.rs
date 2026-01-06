mod commands;

use clap::{Parser, Subcommand};
use anyhow::Result;
use tracing_subscriber;

#[derive(Parser)]
#[command(name = "ade-cli")]
#[command(about = "Ade Sidechain CLI Tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(short, long, default_value = "http://localhost:8899")]
    rpc_url: String,

    #[arg(short, long)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Genesis block generation and management
    Genesis(commands::genesis::GenesisArgs),
    /// Node management commands
    Node {
        #[command(subcommand)]
        action: NodeAction,
    },
    /// Account operations
    Account {
        #[command(subcommand)]
        action: AccountAction,
    },
    /// Transaction operations
    Transaction {
        #[command(subcommand)]
        action: TransactionAction,
    },
    /// Validator operations
    Validator {
        #[command(subcommand)]
        action: ValidatorAction,
    },
    /// AI Agent operations
    Agent {
        #[command(subcommand)]
        action: AgentAction,
    },
    /// Bridge operations
    Bridge {
        #[command(subcommand)]
        action: BridgeAction,
    },
}

#[derive(Subcommand)]
enum NodeAction {
    /// Get node info
    Info,
    /// Get current slot
    Slot,
    /// Get block height
    Height,
    /// Get node health
    Health,
    /// Get node metrics
    Metrics,
}

#[derive(Subcommand)]
enum AccountAction {
    /// Get account balance
    Balance {
        #[arg(value_name = "ADDRESS")]
        address: String,
    },
    /// Get account info
    Info {
        #[arg(value_name = "ADDRESS")]
        address: String,
    },
    /// Request airdrop
    Airdrop {
        #[arg(value_name = "ADDRESS")]
        address: String,
        #[arg(value_name = "AMOUNT")]
        amount: u64,
    },
}

#[derive(Subcommand)]
enum TransactionAction {
    /// Get transaction
    Get {
        #[arg(value_name = "SIGNATURE")]
        signature: String,
    },
    /// Get transaction count
    Count,
    /// Get signatures for address
    Signatures {
        #[arg(value_name = "ADDRESS")]
        address: String,
        #[arg(long, default_value = "10")]
        limit: usize,
    },
}

#[derive(Subcommand)]
enum ValidatorAction {
    /// List validators
    List,
    /// Get validator info
    Info {
        #[arg(value_name = "PUBKEY")]
        pubkey: String,
    },
    /// Get leader schedule
    Schedule,
}

#[derive(Subcommand)]
enum AgentAction {
    /// Deploy AI agent
    Deploy {
        #[arg(long)]
        model_hash: String,
        #[arg(long)]
        config_file: String,
    },
    /// Execute AI agent
    Execute {
        #[arg(value_name = "AGENT_ID")]
        agent_id: String,
        #[arg(long)]
        input_file: String,
        #[arg(long, default_value = "100000")]
        max_compute: u64,
    },
    /// Get agent info
    Info {
        #[arg(value_name = "AGENT_ID")]
        agent_id: String,
    },
    /// List agents
    List {
        #[arg(long)]
        owner: Option<String>,
    },
}

#[derive(Subcommand)]
enum BridgeAction {
    /// Initiate deposit
    Deposit {
        #[arg(long)]
        from_chain: String,
        #[arg(long)]
        amount: u64,
        #[arg(long)]
        token: String,
    },
    /// Initiate withdrawal
    Withdraw {
        #[arg(long)]
        to_chain: String,
        #[arg(long)]
        amount: u64,
        #[arg(long)]
        recipient: String,
    },
    /// Get bridge status
    Status {
        #[arg(value_name = "ID")]
        id: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.verbose {
        tracing_subscriber::fmt::init();
    }

    match cli.command {
        Commands::Genesis(args) => {
            commands::genesis::execute(args).await?;
        }
        Commands::Node { action } => {
            commands::node::handle_node_command(action, &cli.rpc_url).await?;
        }
        Commands::Account { action } => {
            commands::account::handle_account_command(action, &cli.rpc_url).await?;
        }
        Commands::Transaction { action } => {
            commands::transaction::handle_transaction_command(action, &cli.rpc_url).await?;
        }
        Commands::Validator { action } => {
            commands::validator::handle_validator_command(action, &cli.rpc_url).await?;
        }
        Commands::Agent { action } => {
            commands::agent::handle_agent_command(action, &cli.rpc_url).await?;
        }
        Commands::Bridge { action } => {
            commands::bridge::handle_bridge_command(action, &cli.rpc_url).await?;
        }
    }

    Ok(())
}






