use anyhow::Result;
use colored::*;
use crate::commands::rpc::RpcClient;

pub async fn handle_validator_command(action: crate::ValidatorAction, rpc_url: &str) -> Result<()> {
    let client = RpcClient::new(rpc_url);

    match action {
        crate::ValidatorAction::List => {
            let result = client.call("getValidators", None).await?;
            
            if let Some(current) = result.get("current").and_then(|c| c.as_array()) {
                println!("{}: {}", "Active Validators".bold().green(), current.len());
                
                for (i, validator) in current.iter().enumerate() {
                    if let (Some(pubkey), Some(stake)) = (
                        validator.get("identityPubkey"),
                        validator.get("stake")
                    ) {
                        println!("  {}. {} - Stake: {}", i + 1, pubkey, stake);
                    }
                }
            }
        }
        crate::ValidatorAction::Info { pubkey } => {
            println!("{}: {}", "Validator".bold().green(), pubkey);
            // Would query specific validator info
        }
        crate::ValidatorAction::Schedule => {
            let schedule = client.call("getLeaderSchedule", None).await?;
            println!("{}", "Leader Schedule".bold().green());
            println!("{}", serde_json::to_string_pretty(&schedule)?);
        }
    }

    Ok(())
}

