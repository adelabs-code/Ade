use anyhow::Result;
use colored::*;
use crate::commands::rpc::RpcClient;

pub async fn handle_node_command(action: crate::NodeAction, rpc_url: &str) -> Result<()> {
    let client = RpcClient::new(rpc_url);

    match action {
        crate::NodeAction::Info => {
            let version = client.call("getVersion", None).await?;
            println!("{}", "Node Information".bold().green());
            println!("Version: {}", version);
            
            let identity = client.call("getIdentity", None).await?;
            println!("Identity: {}", identity);
        }
        crate::NodeAction::Slot => {
            let slot = client.call("getSlot", None).await?;
            println!("{}: {}", "Current Slot".bold(), slot);
        }
        crate::NodeAction::Height => {
            let height = client.call("getBlockHeight", None).await?;
            println!("{}: {}", "Block Height".bold(), height);
        }
        crate::NodeAction::Health => {
            match reqwest::get(format!("{}/health", rpc_url)).await {
                Ok(response) if response.status().is_success() => {
                    println!("{}", "✓ Node is healthy".green());
                }
                _ => {
                    println!("{}", "✗ Node is unhealthy".red());
                }
            }
        }
        crate::NodeAction::Metrics => {
            match reqwest::get(format!("{}/metrics", rpc_url)).await {
                Ok(response) => {
                    let metrics: serde_json::Value = response.json().await?;
                    println!("{}", "Node Metrics".bold().green());
                    println!("{}", serde_json::to_string_pretty(&metrics)?);
                }
                Err(e) => {
                    println!("{}: {}", "Error getting metrics".red(), e);
                }
            }
        }
    }

    Ok(())
}






