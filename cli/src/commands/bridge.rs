use anyhow::Result;
use colored::*;
use serde_json::json;
use crate::commands::rpc::RpcClient;

pub async fn handle_bridge_command(action: crate::BridgeAction, rpc_url: &str) -> Result<()> {
    let client = RpcClient::new(rpc_url);

    match action {
        crate::BridgeAction::Deposit { from_chain, amount, token } => {
            println!("Initiating bridge deposit...");
            
            let result = client.call("bridgeDeposit", Some(json!({
                "fromChain": from_chain,
                "amount": amount,
                "tokenAddress": token
            }))).await?;
            
            println!("{}", "Deposit initiated".green());
            println!("{}: {}", "Deposit ID".bold(), result.get("depositId").unwrap());
            println!("{}: {}", "Signature".bold(), result.get("signature").unwrap());
        }
        crate::BridgeAction::Withdraw { to_chain, amount, recipient } => {
            println!("Initiating bridge withdrawal...");
            
            let result = client.call("bridgeWithdraw", Some(json!({
                "toChain": to_chain,
                "amount": amount,
                "recipient": recipient
            }))).await?;
            
            println!("{}", "Withdrawal initiated".green());
            println!("{}: {}", "Withdrawal ID".bold(), result.get("withdrawalId").unwrap());
            println!("{}: {}", "Signature".bold(), result.get("signature").unwrap());
        }
        crate::BridgeAction::Status { id } => {
            let result = client.call("getBridgeStatus", Some(json!({ "id": id }))).await?;
            
            println!("{}", "Bridge Operation Status".bold().green());
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
    }

    Ok(())
}

