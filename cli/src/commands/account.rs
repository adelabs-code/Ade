use anyhow::Result;
use colored::*;
use serde_json::json;
use crate::commands::rpc::RpcClient;

pub async fn handle_account_command(action: crate::AccountAction, rpc_url: &str) -> Result<()> {
    let client = RpcClient::new(rpc_url);

    match action {
        crate::AccountAction::Balance { address } => {
            let result = client.call("getBalance", Some(json!({ "address": address }))).await?;
            
            if let Some(value) = result.get("value") {
                let lamports = value.as_u64().unwrap_or(0);
                println!("{}: {} lamports ({} ADE)", 
                    "Balance".bold().green(),
                    lamports,
                    lamports as f64 / 1_000_000_000.0
                );
            }
        }
        crate::AccountAction::Info { address } => {
            let result = client.call("getAccountInfo", Some(json!({ "address": address }))).await?;
            
            println!("{}", "Account Information".bold().green());
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        crate::AccountAction::Airdrop { address, amount } => {
            println!("Requesting airdrop of {} lamports to {}...", amount, address);
            
            let signature = client.call("requestAirdrop", Some(json!({
                "address": address,
                "lamports": amount
            }))).await?;
            
            println!("{}: {}", "Airdrop requested".green(), signature);
        }
    }

    Ok(())
}

