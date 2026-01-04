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

    /// Execute bridge deposit instruction - mints wrapped tokens to recipient
    /// 
    /// This function:
    /// 1. Creates or updates the wrapped token mint for the bridged asset
    /// 2. Creates or updates the recipient's token account
    /// 3. Mints wrapped tokens to the recipient
    /// 4. Records the deposit for relayer verification
    fn execute_bridge_deposit(
        &self,
        from_chain: &str,
        amount: u64,
        token_address: &[u8],
        accounts: &[AccountMeta],
        context: &mut ExecutionContext,
    ) -> Result<ExecutionResult> {
        if accounts.is_empty() {
            return Err(anyhow::anyhow!("No accounts provided for bridge deposit"));
        }

        // Validate amount
        if amount == 0 {
            return Err(anyhow::anyhow!("Deposit amount cannot be zero"));
        }

        let recipient_key = &accounts[0].pubkey;
        context.log(format!(
            "Bridge deposit from {} of {} tokens to {}",
            from_chain,
            amount,
            bs58::encode(recipient_key).into_string()
        ));

        let mut accounts_map = self.accounts.write().unwrap();
        let mut modified_accounts = Vec::new();

        // 1. Get or create wrapped token mint account
        let wrapped_mint_key = self.derive_wrapped_mint_key(from_chain, token_address);
        let wrapped_mint = accounts_map.entry(wrapped_mint_key.clone())
            .or_insert_with(|| {
                let mut account = Account::new(0, vec![]);
                // Initialize mint data
                let mint_data = MintAccountData {
                    mint_authority: Some(self.derive_bridge_authority_key().to_vec()),
                    supply: 0,
                    decimals: 9, // Default decimals
                    is_initialized: true,
                    freeze_authority: Some(self.derive_bridge_authority_key().to_vec()),
                };
                account.data = bincode::serialize(&mint_data).unwrap_or_default();
                account
            });
        
        // Update mint supply
        if let Ok(mut mint_data) = bincode::deserialize::<MintAccountData>(&wrapped_mint.data) {
            mint_data.supply = mint_data.supply.saturating_add(amount);
            wrapped_mint.data = bincode::serialize(&mint_data)?;
            modified_accounts.push(wrapped_mint_key.clone());
            context.log(format!("Updated mint supply: {}", mint_data.supply));
        }

        // 2. Get or create recipient's token account for this wrapped token
        let recipient_token_key = self.derive_wrapped_token_key(token_address, recipient_key);
        let recipient_token_account = accounts_map.entry(recipient_token_key.clone())
            .or_insert_with(|| {
                let mut account = Account::new(0, vec![]);
                let token_data = TokenAccountData::new(
                    wrapped_mint_key.clone(),
                    recipient_key.clone(),
                );
                account.data = bincode::serialize(&token_data).unwrap_or_default();
                account.owner = wrapped_mint_key.clone(); // Token program owns the account
                account
            });
        
        // 3. Update token account balance (mint tokens)
        if let Ok(mut token_data) = bincode::deserialize::<TokenAccountData>(&recipient_token_account.data) {
            // Check if account is frozen
            if token_data.is_frozen() {
                return Err(anyhow::anyhow!("Recipient token account is frozen"));
            }
            
            // Verify ownership
            if token_data.owner != *recipient_key {
                return Err(anyhow::anyhow!("Token account owner mismatch"));
            }
            
            // Mint tokens
            token_data.amount = token_data.amount.saturating_add(amount);
            recipient_token_account.data = bincode::serialize(&token_data)?;
            modified_accounts.push(recipient_token_key.clone());
            
            context.log(format!("Minted {} wrapped tokens to token account", amount));
            context.log(format!("New token balance: {}", token_data.amount));
        } else {
            // Fallback: create new token data
            let token_data = TokenAccountData::with_amount(
                wrapped_mint_key.clone(),
                recipient_key.clone(),
                amount,
            );
            recipient_token_account.data = bincode::serialize(&token_data)?;
            modified_accounts.push(recipient_token_key.clone());
            context.log(format!("Created new token account with {} tokens", amount));
        }

        // 4. Also credit lamports for gas (optional: bridge fee could cover this)
        let recipient_account = accounts_map.entry(recipient_key.clone())
            .or_insert_with(|| Account::new(0, vec![]));
        modified_accounts.push(recipient_key.clone());

        // 5. Record the deposit
        let deposit_record_key = self.derive_deposit_record_key(token_address, context.slot);
        let deposit_record = accounts_map.entry(deposit_record_key.clone())
            .or_insert_with(|| Account::new(0, vec![]));
        
        deposit_record.data = bincode::serialize(&BridgeDepositRecord {
            from_chain: from_chain.to_string(),
            token_address: token_address.to_vec(),
            amount,
            recipient: recipient_key.clone(),
            slot: context.slot,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })?;
        modified_accounts.push(deposit_record_key);

        context.log("Bridge deposit completed successfully".to_string());
        
        Ok(ExecutionResult {
            success: true,
            compute_units_consumed: 5000,
            logs: context.logs.clone(),
            error: None,
            modified_accounts,
        })
    }

    /// Execute bridge withdraw instruction - burns wrapped tokens and initiates unlock
    /// 
    /// This function:
    /// 1. Validates the sender's token account
    /// 2. Burns wrapped tokens from the sender's token account
    /// 3. Updates the mint supply
    /// 4. Creates a withdrawal record for the relayer
    fn execute_bridge_withdraw(
        &self,
        to_chain: &str,
        amount: u64,
        recipient: &[u8],
        accounts: &[AccountMeta],
        context: &mut ExecutionContext,
    ) -> Result<ExecutionResult> {
        if accounts.len() < 2 {
            return Err(anyhow::anyhow!("Insufficient accounts: need sender and token account"));
        }

        // Validate amount
        if amount == 0 {
            return Err(anyhow::anyhow!("Withdrawal amount cannot be zero"));
        }

        let sender_key = &accounts[0].pubkey;
        let token_account_key = if accounts.len() > 1 { 
            &accounts[1].pubkey 
        } else { 
            sender_key 
        };
        
        context.log(format!(
            "Bridge withdraw to {} of {} tokens from {}",
            to_chain,
            amount,
            bs58::encode(sender_key).into_string()
        ));

        let mut accounts_map = self.accounts.write().unwrap();
        let mut modified_accounts = Vec::new();

        // 1. Get and validate the sender's token account
        let sender_token_account = accounts_map.get_mut(token_account_key)
            .ok_or_else(|| anyhow::anyhow!("Sender token account not found"))?;

        // Try to parse as TokenAccountData
        if let Ok(mut token_data) = bincode::deserialize::<TokenAccountData>(&sender_token_account.data) {
            // Verify ownership
            if token_data.owner != *sender_key {
                return Err(anyhow::anyhow!("Token account owner mismatch"));
            }
            
            // Check if account is frozen
            if token_data.is_frozen() {
                return Err(anyhow::anyhow!("Token account is frozen"));
            }
            
            // Check sufficient balance
            if token_data.amount < amount {
                return Err(anyhow::anyhow!(
                    "Insufficient token balance: {} < {}",
                    token_data.amount,
                    amount
                ));
            }
            
            // 2. Burn tokens from account
            token_data.amount = token_data.amount.saturating_sub(amount);
            sender_token_account.data = bincode::serialize(&token_data)?;
            modified_accounts.push(token_account_key.clone());
            
            context.log(format!("Burned {} tokens from token account", amount));
            context.log(format!("Remaining token balance: {}", token_data.amount));
            
            // 3. Update mint supply (decrease by burned amount)
            if let Some(mint_account) = accounts_map.get_mut(&token_data.mint) {
                if let Ok(mut mint_data) = bincode::deserialize::<MintAccountData>(&mint_account.data) {
                    mint_data.supply = mint_data.supply.saturating_sub(amount);
                    mint_account.data = bincode::serialize(&mint_data)?;
                    modified_accounts.push(token_data.mint.clone());
                    context.log(format!("Updated mint supply: {}", mint_data.supply));
                }
            }
        } else {
            // Fallback: treat as lamports (legacy behavior)
            let sender_account = accounts_map.get_mut(sender_key)
                .ok_or_else(|| anyhow::anyhow!("Sender account not found"))?;

            if sender_account.lamports < amount {
                return Err(anyhow::anyhow!(
                    "Insufficient balance for withdrawal: {} < {}",
                    sender_account.lamports,
                    amount
                ));
            }

            sender_account.lamports -= amount;
            modified_accounts.push(sender_key.clone());
            
            context.log(format!("Burned {} lamports from sender (legacy mode)", amount));
        }

        // 4. Create withdrawal record for the relayer to process
        let withdrawal_record_key = self.derive_withdrawal_record_key(sender_key, context.slot);
        let mut withdrawal_record = Account::new(0, vec![]);
        
        withdrawal_record.data = bincode::serialize(&BridgeWithdrawalRecord {
            to_chain: to_chain.to_string(),
            amount,
            sender: sender_key.clone(),
            recipient: recipient.to_vec(),
            slot: context.slot,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            status: WithdrawalStatus::Pending,
        })?;
        
        accounts_map.insert(withdrawal_record_key.clone(), withdrawal_record);
        modified_accounts.push(withdrawal_record_key);

        context.log(format!(
            "Withdrawal record created for {} on {}",
            bs58::encode(recipient).into_string(),
            to_chain
        ));
        
        Ok(ExecutionResult {
            success: true,
            compute_units_consumed: 5000,
            logs: context.logs.clone(),
            error: None,
            modified_accounts,
        })
    }

    /// Derive wrapped token account key
    fn derive_wrapped_token_key(&self, token_address: &[u8], owner: &[u8]) -> Vec<u8> {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(b"wrapped_token");
        hasher.update(token_address);
        hasher.update(owner);
        hasher.finalize().to_vec()
    }

    /// Derive bridge escrow account key
    fn derive_bridge_escrow_key(&self, chain: &str) -> Vec<u8> {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(b"bridge_escrow");
        hasher.update(chain.as_bytes());
        hasher.finalize().to_vec()
    }

    /// Derive deposit record key
    fn derive_deposit_record_key(&self, token_address: &[u8], slot: u64) -> Vec<u8> {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(b"deposit_record");
        hasher.update(token_address);
        hasher.update(&slot.to_le_bytes());
        hasher.finalize().to_vec()
    }

    /// Derive withdrawal record key
    fn derive_withdrawal_record_key(&self, sender: &[u8], slot: u64) -> Vec<u8> {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(b"withdrawal_record");
        hasher.update(sender);
        hasher.update(&slot.to_le_bytes());
        hasher.finalize().to_vec()
    }

    /// Derive wrapped token mint key for a specific chain and token
    fn derive_wrapped_mint_key(&self, from_chain: &str, token_address: &[u8]) -> Vec<u8> {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(b"wrapped_mint");
        hasher.update(from_chain.as_bytes());
        hasher.update(token_address);
        hasher.finalize().to_vec()
    }

    /// Derive the bridge authority key (PDA-like)
    fn derive_bridge_authority_key(&self) -> Vec<u8> {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(b"bridge_authority");
        hasher.update(b"ADE_BRIDGE_PROGRAM");
        hasher.finalize().to_vec()
    }
}

/// Record of a bridge deposit
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BridgeDepositRecord {
    pub from_chain: String,
    pub token_address: Vec<u8>,
    pub amount: u64,
    pub recipient: Vec<u8>,
    pub slot: u64,
    pub timestamp: u64,
}

/// SPL Token Account structure for proper token handling
/// This mirrors the Solana SPL Token program's account layout
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TokenAccountData {
    /// The mint/token address this account holds
    pub mint: Vec<u8>,
    /// Owner of this token account
    pub owner: Vec<u8>,
    /// Token balance
    pub amount: u64,
    /// Optional delegate
    pub delegate: Option<Vec<u8>>,
    /// Account state (0=uninitialized, 1=initialized, 2=frozen)
    pub state: u8,
    /// Is this a native token account (SOL)?
    pub is_native: bool,
    /// Delegated amount
    pub delegated_amount: u64,
    /// Optional close authority
    pub close_authority: Option<Vec<u8>>,
}

impl TokenAccountData {
    /// Create a new token account
    pub fn new(mint: Vec<u8>, owner: Vec<u8>) -> Self {
        Self {
            mint,
            owner,
            amount: 0,
            delegate: None,
            state: 1, // Initialized
            is_native: false,
            delegated_amount: 0,
            close_authority: None,
        }
    }
    
    /// Create a new token account with initial amount
    pub fn with_amount(mint: Vec<u8>, owner: Vec<u8>, amount: u64) -> Self {
        let mut account = Self::new(mint, owner);
        account.amount = amount;
        account
    }
    
    /// Check if account is frozen
    pub fn is_frozen(&self) -> bool {
        self.state == 2
    }
    
    /// Freeze the account
    pub fn freeze(&mut self) {
        self.state = 2;
    }
    
    /// Unfreeze the account
    pub fn unfreeze(&mut self) {
        self.state = 1;
    }
}

/// Mint account structure for token minting authority
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MintAccountData {
    /// Mint authority (can mint new tokens)
    pub mint_authority: Option<Vec<u8>>,
    /// Total supply
    pub supply: u64,
    /// Number of decimals
    pub decimals: u8,
    /// Is initialized
    pub is_initialized: bool,
    /// Freeze authority (can freeze accounts)
    pub freeze_authority: Option<Vec<u8>>,
}

/// Record of a bridge withdrawal
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BridgeWithdrawalRecord {
    pub to_chain: String,
    pub amount: u64,
    pub sender: Vec<u8>,
    pub recipient: Vec<u8>,
    pub slot: u64,
    pub timestamp: u64,
    pub status: WithdrawalStatus,
}

/// Status of a bridge withdrawal
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum WithdrawalStatus {
    Pending,
    Processing,
    Completed,
    Failed,
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





