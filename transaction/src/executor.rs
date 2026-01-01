use anyhow::{Result, Context};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::{info, warn, error};

use crate::instruction::{Instruction, InstructionType, AccountMeta};
use crate::account::{Account, AccountState};
use crate::transaction::Transaction;

/// Instruction execution result
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub success: bool,
    pub compute_units_consumed: u64,
    pub logs: Vec<String>,
    pub error: Option<String>,
    pub modified_accounts: Vec<Vec<u8>>,
}

/// Account changes during execution
#[derive(Debug, Clone)]
pub struct AccountChange {
    pub address: Vec<u8>,
    pub pre_balance: u64,
    pub post_balance: u64,
    pub data_changed: bool,
}

/// Transaction execution context
pub struct ExecutionContext {
    pub slot: u64,
    pub compute_budget: u64,
    pub compute_consumed: u64,
    pub logs: Vec<String>,
}

impl ExecutionContext {
    pub fn new(slot: u64, compute_budget: u64) -> Self {
        Self {
            slot,
            compute_budget,
            compute_consumed: 0,
            logs: Vec::new(),
        }
    }

    pub fn consume_compute(&mut self, units: u64) -> Result<()> {
        self.compute_consumed += units;
        if self.compute_consumed > self.compute_budget {
            return Err(anyhow::anyhow!(
                "Compute budget exceeded: {} > {}",
                self.compute_consumed,
                self.compute_budget
            ));
        }
        Ok(())
    }

    pub fn log(&mut self, message: String) {
        self.logs.push(message);
    }
}

/// Instruction executor
pub struct InstructionExecutor {
    accounts: Arc<RwLock<HashMap<Vec<u8>, Account>>>,
    compute_costs: HashMap<String, u64>,
}

impl InstructionExecutor {
    pub fn new(accounts: Arc<RwLock<HashMap<Vec<u8>, Account>>>) -> Self {
        let mut compute_costs = HashMap::new();
        
        // Define compute costs for each instruction type
        compute_costs.insert("Transfer".to_string(), 150);
        compute_costs.insert("CreateAccount".to_string(), 500);
        compute_costs.insert("AIAgentDeploy".to_string(), 50000);
        compute_costs.insert("AIAgentExecute".to_string(), 100000);
        compute_costs.insert("AIAgentUpdate".to_string(), 25000);
        compute_costs.insert("BridgeDeposit".to_string(), 5000);
        compute_costs.insert("BridgeWithdraw".to_string(), 5000);
        
        Self {
            accounts,
            compute_costs,
        }
    }

    /// Execute a single instruction
    pub fn execute_instruction(
        &self,
        instruction: &Instruction,
        context: &mut ExecutionContext,
    ) -> Result<ExecutionResult> {
        // Parse instruction type
        let ix_type = instruction.parse_data()
            .context("Failed to parse instruction data")?;

        // Get compute cost
        let type_name = match &ix_type {
            InstructionType::Transfer { .. } => "Transfer",
            InstructionType::CreateAccount { .. } => "CreateAccount",
            InstructionType::AIAgentDeploy { .. } => "AIAgentDeploy",
            InstructionType::AIAgentExecute { .. } => "AIAgentExecute",
            InstructionType::AIAgentUpdate { .. } => "AIAgentUpdate",
            InstructionType::BridgeDeposit { .. } => "BridgeDeposit",
            InstructionType::BridgeWithdraw { .. } => "BridgeWithdraw",
        };

        let compute_cost = self.compute_costs.get(type_name).cloned().unwrap_or(1000);
        context.consume_compute(compute_cost)?;
        context.log(format!("Executing instruction: {}", type_name));

        // Execute based on type
        let result = match ix_type {
            InstructionType::Transfer { from, to, amount } => {
                self.execute_transfer(&from, &to, amount, &instruction.accounts, context)?
            }
            InstructionType::CreateAccount { owner, space, lamports } => {
                self.execute_create_account(&owner, space, lamports, &instruction.accounts, context)?
            }
            InstructionType::AIAgentDeploy { agent_id, model_hash, config } => {
                self.execute_ai_agent_deploy(&agent_id, &model_hash, &config, &instruction.accounts, context)?
            }
            InstructionType::AIAgentExecute { agent_id, input_data, max_compute } => {
                self.execute_ai_agent_execute(&agent_id, &input_data, max_compute, &instruction.accounts, context)?
            }
            InstructionType::AIAgentUpdate { agent_id, new_model_hash, new_config } => {
                self.execute_ai_agent_update(&agent_id, &new_model_hash, &new_config, &instruction.accounts, context)?
            }
            InstructionType::BridgeDeposit { from_chain, amount, token_address } => {
                self.execute_bridge_deposit(&from_chain, amount, &token_address, &instruction.accounts, context)?
            }
            InstructionType::BridgeWithdraw { to_chain, amount, recipient } => {
                self.execute_bridge_withdraw(&to_chain, amount, &recipient, &instruction.accounts, context)?
            }
        };

        Ok(result)
    }

    /// Execute transfer instruction
    fn execute_transfer(
        &self,
        from: &[u8],
        to: &[u8],
        amount: u64,
        accounts: &[AccountMeta],
        context: &mut ExecutionContext,
    ) -> Result<ExecutionResult> {
        let mut accounts_map = self.accounts.write().unwrap();
        
        // Get source account
        let from_account = accounts_map.get_mut(from)
            .ok_or_else(|| anyhow::anyhow!("Source account not found"))?;
        
        // Check balance
        if from_account.lamports < amount {
            return Err(anyhow::anyhow!(
                "Insufficient balance: {} < {}",
                from_account.lamports,
                amount
            ));
        }
        
        // Deduct from source
        from_account.lamports -= amount;
        context.log(format!("Debited {} from source", amount));
        
        // Get or create destination account
        let to_account = accounts_map.entry(to.to_vec())
            .or_insert_with(|| Account::new(0, vec![]));
        
        // Credit to destination
        to_account.lamports += amount;
        context.log(format!("Credited {} to destination", amount));
        
        Ok(ExecutionResult {
            success: true,
            compute_units_consumed: 150,
            logs: context.logs.clone(),
            error: None,
            modified_accounts: vec![from.to_vec(), to.to_vec()],
        })
    }

    /// Execute create account instruction
    fn execute_create_account(
        &self,
        owner: &[u8],
        space: u64,
        lamports: u64,
        accounts: &[AccountMeta],
        context: &mut ExecutionContext,
    ) -> Result<ExecutionResult> {
        if accounts.len() < 2 {
            return Err(anyhow::anyhow!("Insufficient accounts"));
        }
        
        let new_account_key = &accounts[1].pubkey;
        let mut accounts_map = self.accounts.write().unwrap();
        
        // Check if account already exists
        if accounts_map.contains_key(new_account_key) {
            return Err(anyhow::anyhow!("Account already exists"));
        }
        
        // Create new account
        let mut account = Account::new(lamports, owner.to_vec());
        account.data = vec![0u8; space as usize];
        
        accounts_map.insert(new_account_key.clone(), account);
        context.log(format!("Created account with {} lamports and {} bytes", lamports, space));
        
        Ok(ExecutionResult {
            success: true,
            compute_units_consumed: 500,
            logs: context.logs.clone(),
            error: None,
            modified_accounts: vec![new_account_key.clone()],
        })
    }

    /// Execute AI agent deploy instruction
    fn execute_ai_agent_deploy(
        &self,
        agent_id: &[u8],
        model_hash: &[u8],
        config: &[u8],
        accounts: &[AccountMeta],
        context: &mut ExecutionContext,
    ) -> Result<ExecutionResult> {
        let mut accounts_map = self.accounts.write().unwrap();
        
        // Check if agent already exists
        if accounts_map.contains_key(agent_id) {
            return Err(anyhow::anyhow!("AI agent already exists"));
        }
        
        // Get owner account
        let owner_key = &accounts[0].pubkey;
        
        // Create agent account
        let mut agent_account = Account::new(0, owner_key.clone());
        agent_account.data = bincode::serialize(&AccountState::AIAgent {
            agent_id: agent_id.to_vec(),
            model_hash: model_hash.to_vec(),
            owner: owner_key.clone(),
            execution_count: 0,
            total_compute_used: 0,
        })?;
        
        accounts_map.insert(agent_id.to_vec(), agent_account);
        context.log(format!("Deployed AI agent: {:?}", bs58::encode(agent_id).into_string()));
        
        Ok(ExecutionResult {
            success: true,
            compute_units_consumed: 50000,
            logs: context.logs.clone(),
            error: None,
            modified_accounts: vec![agent_id.to_vec()],
        })
    }

    /// Execute AI agent execute instruction
    fn execute_ai_agent_execute(
        &self,
        agent_id: &[u8],
        input_data: &[u8],
        max_compute: u64,
        accounts: &[AccountMeta],
        context: &mut ExecutionContext,
    ) -> Result<ExecutionResult> {
        // Consume compute units
        context.consume_compute(max_compute)?;
        
        let mut accounts_map = self.accounts.write().unwrap();
        
        // Get agent account
        let agent_account = accounts_map.get_mut(agent_id)
            .ok_or_else(|| anyhow::anyhow!("AI agent not found"))?;
        
        // Update agent state
        if let Ok(mut state) = bincode::deserialize::<AccountState>(&agent_account.data) {
            if let AccountState::AIAgent { 
                ref mut execution_count, 
                ref mut total_compute_used,
                .. 
            } = state {
                *execution_count += 1;
                *total_compute_used += max_compute;
                
                agent_account.data = bincode::serialize(&state)?;
            }
        }
        
        context.log(format!("Executed AI agent with {} compute units", max_compute));
        
        Ok(ExecutionResult {
            success: true,
            compute_units_consumed: max_compute,
            logs: context.logs.clone(),
            error: None,
            modified_accounts: vec![agent_id.to_vec()],
        })
    }

    /// Execute AI agent update instruction
    fn execute_ai_agent_update(
        &self,
        agent_id: &[u8],
        new_model_hash: &[u8],
        new_config: &[u8],
        accounts: &[AccountMeta],
        context: &mut ExecutionContext,
    ) -> Result<ExecutionResult> {
        let mut accounts_map = self.accounts.write().unwrap();
        
        // Get agent account
        let agent_account = accounts_map.get_mut(agent_id)
            .ok_or_else(|| anyhow::anyhow!("AI agent not found"))?;
        
        // Verify ownership
        let caller = &accounts[0].pubkey;
        if let Ok(state) = bincode::deserialize::<AccountState>(&agent_account.data) {
            if let AccountState::AIAgent { owner, .. } = state {
                if owner != *caller {
                    return Err(anyhow::anyhow!("Unauthorized: not the owner"));
                }
            }
        }
        
        context.log("Updated AI agent configuration".to_string());
        
        Ok(ExecutionResult {
            success: true,
            compute_units_consumed: 25000,
            logs: context.logs.clone(),
            error: None,
            modified_accounts: vec![agent_id.to_vec()],
        })
    }

    /// Execute bridge deposit instruction
    fn execute_bridge_deposit(
        &self,
        from_chain: &str,
        amount: u64,
        token_address: &[u8],
        accounts: &[AccountMeta],
        context: &mut ExecutionContext,
    ) -> Result<ExecutionResult> {
        context.log(format!("Bridge deposit from {} of {} tokens", from_chain, amount));
        
        // In a real implementation, this would interact with bridge contracts
        
        Ok(ExecutionResult {
            success: true,
            compute_units_consumed: 5000,
            logs: context.logs.clone(),
            error: None,
            modified_accounts: vec![],
        })
    }

    /// Execute bridge withdraw instruction
    fn execute_bridge_withdraw(
        &self,
        to_chain: &str,
        amount: u64,
        recipient: &[u8],
        accounts: &[AccountMeta],
        context: &mut ExecutionContext,
    ) -> Result<ExecutionResult> {
        context.log(format!("Bridge withdraw to {} of {} tokens", to_chain, amount));
        
        // In a real implementation, this would interact with bridge contracts
        
        Ok(ExecutionResult {
            success: true,
            compute_units_consumed: 5000,
            logs: context.logs.clone(),
            error: None,
            modified_accounts: vec![],
        })
    }
}

/// Transaction executor
pub struct TransactionExecutor {
    instruction_executor: InstructionExecutor,
    max_compute_budget: u64,
}

impl TransactionExecutor {
    pub fn new(accounts: Arc<RwLock<HashMap<Vec<u8>, Account>>>) -> Self {
        Self {
            instruction_executor: InstructionExecutor::new(accounts),
            max_compute_budget: 1_400_000,
        }
    }

    /// Execute a transaction
    pub fn execute_transaction(
        &self,
        transaction: &Transaction,
        slot: u64,
    ) -> Result<ExecutionResult> {
        // Verify signatures
        transaction.verify()
            .map_err(|e| anyhow::anyhow!("Signature verification failed: {}", e))?;

        // Create execution context
        let mut context = ExecutionContext::new(slot, self.max_compute_budget);
        
        let mut all_modified_accounts = Vec::new();
        let mut total_compute = 0u64;

        // Execute each instruction
        for (idx, instruction) in transaction.message.instructions.iter().enumerate() {
            context.log(format!("--- Instruction {} ---", idx));
            
            match self.instruction_executor.execute_instruction(instruction, &mut context) {
                Ok(result) => {
                    total_compute += result.compute_units_consumed;
                    all_modified_accounts.extend(result.modified_accounts);
                    
                    if !result.success {
                        return Ok(ExecutionResult {
                            success: false,
                            compute_units_consumed: total_compute,
                            logs: context.logs.clone(),
                            error: result.error,
                            modified_accounts: all_modified_accounts,
                        });
                    }
                }
                Err(e) => {
                    return Ok(ExecutionResult {
                        success: false,
                        compute_units_consumed: total_compute,
                        logs: context.logs.clone(),
                        error: Some(e.to_string()),
                        modified_accounts: all_modified_accounts,
                    });
                }
            }
        }

        Ok(ExecutionResult {
            success: true,
            compute_units_consumed: total_compute,
            logs: context.logs,
            error: None,
            modified_accounts: all_modified_accounts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::Keypair;
    use rand::rngs::OsRng;

    #[test]
    fn test_transfer_execution() {
        let accounts = Arc::new(RwLock::new(HashMap::new()));
        
        // Create source account
        let from_key = vec![1u8; 32];
        accounts.write().unwrap().insert(from_key.clone(), Account::new(1000, vec![]));
        
        let executor = InstructionExecutor::new(accounts);
        let mut context = ExecutionContext::new(0, 1_000_000);
        
        let to_key = vec![2u8; 32];
        let accounts_meta = vec![
            AccountMeta::new(from_key.clone(), true, true),
            AccountMeta::new(to_key.clone(), false, true),
        ];
        
        let result = executor.execute_transfer(&from_key, &to_key, 500, &accounts_meta, &mut context);
        
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.success);
        assert_eq!(result.modified_accounts.len(), 2);
    }

    #[test]
    fn test_insufficient_balance() {
        let accounts = Arc::new(RwLock::new(HashMap::new()));
        
        let from_key = vec![1u8; 32];
        accounts.write().unwrap().insert(from_key.clone(), Account::new(100, vec![]));
        
        let executor = InstructionExecutor::new(accounts);
        let mut context = ExecutionContext::new(0, 1_000_000);
        
        let to_key = vec![2u8; 32];
        let accounts_meta = vec![
            AccountMeta::new(from_key.clone(), true, true),
            AccountMeta::new(to_key.clone(), false, true),
        ];
        
        let result = executor.execute_transfer(&from_key, &to_key, 500, &accounts_meta, &mut context);
        
        assert!(result.is_err());
    }
}

