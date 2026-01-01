mod node;
mod storage;
mod network;
mod validator;

use clap::Parser;
use tracing::{info, error};
use tracing_subscriber;
use anyhow::Result;

#[derive(Parser, Debug)]
#[command(name = "ade-node")]
#[command(about = "Ade Sidechain Node", long_about = None)]
struct Args {
    #[arg(short, long, default_value = "8899")]
    rpc_port: u16,

    #[arg(short, long, default_value = "9900")]
    gossip_port: u16,

    #[arg(short, long, default_value = "./data")]
    data_dir: String,

    #[arg(short, long)]
    validator_keypair: Option<String>,

    #[arg(long)]
    bootstrap_nodes: Vec<String>,

    #[arg(long, default_value = "false")]
    validator_mode: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    
    info!("Starting Ade Sidechain Node");
    info!("RPC Port: {}", args.rpc_port);
    info!("Gossip Port: {}", args.gossip_port);
    info!("Data Directory: {}", args.data_dir);
    info!("Validator Mode: {}", args.validator_mode);

    let config = node::NodeConfig {
        rpc_port: args.rpc_port,
        gossip_port: args.gossip_port,
        data_dir: args.data_dir,
        validator_keypair: args.validator_keypair,
        bootstrap_nodes: args.bootstrap_nodes,
        validator_mode: args.validator_mode,
    };

    let node = node::Node::new(config)?;
    
    if let Err(e) = node.start().await {
        error!("Node failed: {}", e);
        return Err(e);
    }

    Ok(())
}


