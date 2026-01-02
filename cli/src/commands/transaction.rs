use anyhow::Result;
use colored::*;
use serde_json::json;
use crate::commands::rpc::RpcClient;

pub async fn handle_transaction_command(action: crate::TransactionAction, rpc_url: &str) -> Result<()> {
    let client = RpcClient::new(rpc_url);

    match action {
        crate::TransactionAction::Get { signature } => {
            let result = client.call("getTransaction", Some(json!({ "signature": signature }))).await?;
            
            println!("{}", "Transaction Information".bold().green());
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        crate::TransactionAction::Count => {
            let count = client.call("getTransactionCount", None).await?;
            println!("{}: {}", "Total Transactions".bold(), count);
        }
        crate::TransactionAction::Signatures { address, limit } => {
            let result = client.call("getSignaturesForAddress", Some(json!({
                "address": address,
                "limit": limit
            }))).await?;
            
            if let Some(signatures) = result.as_array() {
                println!("{} (showing {}):", "Signatures".bold().green(), signatures.len());
                for sig in signatures {
                    println!("  {}", sig.get("signature").unwrap());
                }
            }
        }
    }

    Ok(())
}

