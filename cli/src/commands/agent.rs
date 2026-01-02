use anyhow::Result;
use colored::*;
use serde_json::json;
use std::fs;
use crate::commands::rpc::RpcClient;

pub async fn handle_agent_command(action: crate::AgentAction, rpc_url: &str) -> Result<()> {
    let client = RpcClient::new(rpc_url);

    match action {
        crate::AgentAction::Deploy { model_hash, config_file } => {
            println!("Deploying AI agent...");
            
            let config_content = fs::read_to_string(&config_file)?;
            let config: serde_json::Value = serde_json::from_str(&config_content)?;
            
            let result = client.call("deployAIAgent", Some(json!({
                "agentId": format!("agent_{}", current_timestamp()),
                "modelHash": model_hash,
                "config": config
            }))).await?;
            
            println!("{}: {}", "Agent deployed".green(), result.get("agentId").unwrap());
            println!("{}: {}", "Signature".bold(), result.get("signature").unwrap());
        }
        crate::AgentAction::Execute { agent_id, input_file, max_compute } => {
            println!("Executing AI agent {}...", agent_id);
            
            let input_content = fs::read_to_string(&input_file)?;
            let input_data: serde_json::Value = serde_json::from_str(&input_content)?;
            
            let result = client.call("executeAIAgent", Some(json!({
                "agentId": agent_id,
                "inputData": input_data,
                "maxCompute": max_compute
            }))).await?;
            
            println!("{}", "Execution completed".green());
            println!("{}: {}", "Execution ID".bold(), result.get("executionId").unwrap());
            println!("{}: {}", "Compute used".bold(), result.get("computeUnits").unwrap());
        }
        crate::AgentAction::Info { agent_id } => {
            let result = client.call("getAIAgentInfo", Some(json!({ "agentId": agent_id }))).await?;
            
            println!("{}", "AI Agent Information".bold().green());
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        crate::AgentAction::List { owner } => {
            let params = if let Some(owner_addr) = owner {
                Some(json!({ "owner": owner_addr }))
            } else {
                None
            };
            
            let result = client.call("listAIAgents", params).await?;
            
            if let Some(agents) = result.get("agents").and_then(|a| a.as_array()) {
                println!("{}: {}", "AI Agents".bold().green(), agents.len());
                
                for agent in agents {
                    println!("  - {} ({})", 
                        agent.get("agentId").unwrap(),
                        agent.get("status").unwrap()
                    );
                }
            }
        }
    }

    Ok(())
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

