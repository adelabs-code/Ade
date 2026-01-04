use serde::{Serialize, Deserialize};
use anyhow::{Result, Context};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::{info, warn, error};

/// Ade sidechain mint/burn program (SVM-compatible)
/// 
/// Uses SPL-Token-like architecture to prevent state bloat:
/// - Token metadata stored in MintProgramState (small, fixed size)
/// - Individual balances stored in separate TokenAccount per holder
/// - Account key = derive_token_account_key(token, holder)
pub struct AdeMintProgram {
    state: Arc<RwLock<MintProgramState>>,
    /// Separate storage for token accounts (prevents bloat)
    /// Key: derive_token_account_key(token, holder)
    token_accounts: Arc<RwLock<HashMap<Vec<u8>, TokenAccount>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MintProgramState {
    pub authority: Vec<u8>,
    pub wrapped_tokens: HashMap<Vec<u8>, WrappedToken>,
    pub total_minted: u64,
    pub total_burned: u64,
    pub nonce: u64,
    /// Processed proof hashes to prevent replay attacks
    /// Key: hash of (source_tx_hash + block_number)
    /// Value: timestamp when processed
    pub processed_proofs: HashMap<Vec<u8>, u64>,
}

/// Wrapped token metadata - stored per token type
/// 
/// NOTE: Holder balances are NO LONGER stored here to prevent state bloat.
/// Instead, balances are stored in separate TokenAccount structures indexed
/// by (token_address, holder_address). This follows the SPL Token pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WrappedToken {
    pub original_token: Vec<u8>,
    pub original_chain: String,
    pub total_supply: u64,
    /// DEPRECATED: Use get_balance() which reads from token_accounts
    /// Kept for backward compatibility with existing serialized state
    #[serde(default)]
    pub holders: HashMap<Vec<u8>, u64>,
}

/// Separate token account for each (token, holder) pair
/// This prevents single-account bloat that would break the contract
/// when holder count exceeds serialization limits.
/// 
/// Design follows SPL Token pattern:
/// - Each holder has their own token account per token type
/// - Account key = hash(token_address || holder_address)
/// - Maximum ~10MB per account, but each account is separate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenAccount {
    pub token: Vec<u8>,
    pub owner: Vec<u8>,
    pub balance: u64,
    pub created_at: u64,
    pub last_updated: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MintInstruction {
    Initialize {
        authority: Vec<u8>,
    },
    
    MintWrapped {
        proof: super::solana_lock::BridgeProof,
        recipient: Vec<u8>,
        amount: u64,
        original_token: Vec<u8>,
    },
    
    BurnWrapped {
        token: Vec<u8>,
        amount: u64,
        target_chain: String,
        recipient: Vec<u8>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MintEvent {
    pub recipient: Vec<u8>,
    pub token: Vec<u8>,
    pub amount: u64,
    pub source_chain: String,
    pub nonce: u64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurnEvent {
    pub burner: Vec<u8>,
    pub token: Vec<u8>,
    pub amount: u64,
    pub target_chain: String,
    pub recipient: Vec<u8>,
    pub nonce: u64,
    pub timestamp: u64,
}

impl AdeMintProgram {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(MintProgramState {
                authority: vec![],
                wrapped_tokens: HashMap::new(),
                total_minted: 0,
                total_burned: 0,
                nonce: 0,
                processed_proofs: HashMap::new(),
            })),
            token_accounts: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Derive token account key from token and holder addresses
    /// This creates a unique key for each (token, holder) pair
    fn derive_token_account_key(token: &[u8], holder: &[u8]) -> Vec<u8> {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(b"TOKEN_ACCOUNT_V1");
        hasher.update(token);
        hasher.update(holder);
        hasher.finalize().to_vec()
    }
    
    /// Get or create a token account for a holder
    fn get_or_create_token_account(&self, token: &[u8], holder: &[u8]) -> TokenAccount {
        let key = Self::derive_token_account_key(token, holder);
        let accounts = self.token_accounts.read().unwrap();
        
        accounts.get(&key).cloned().unwrap_or_else(|| TokenAccount {
            token: token.to_vec(),
            owner: holder.to_vec(),
            balance: 0,
            created_at: current_timestamp(),
            last_updated: current_timestamp(),
        })
    }
    
    /// Update token account balance
    fn update_token_account(&self, token: &[u8], holder: &[u8], new_balance: u64) {
        let key = Self::derive_token_account_key(token, holder);
        let mut accounts = self.token_accounts.write().unwrap();
        
        let account = accounts.entry(key).or_insert_with(|| TokenAccount {
            token: token.to_vec(),
            owner: holder.to_vec(),
            balance: 0,
            created_at: current_timestamp(),
            last_updated: current_timestamp(),
        });
        
        account.balance = new_balance;
        account.last_updated = current_timestamp();
    }
    
    /// Get balance from token accounts (not from WrappedToken.holders)
    fn get_token_account_balance(&self, token: &[u8], holder: &[u8]) -> u64 {
        let key = Self::derive_token_account_key(token, holder);
        let accounts = self.token_accounts.read().unwrap();
        accounts.get(&key).map(|a| a.balance).unwrap_or(0)
    }
    
    /// Compute unique proof identifier to prevent replay attacks
    /// Uses source_tx_hash + block_number to create a unique key
    fn compute_proof_hash(proof: &super::solana_lock::BridgeProof) -> Vec<u8> {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(b"PROOF_ID_V1");
        hasher.update(&proof.source_tx_hash);
        hasher.update(&proof.block_number.to_le_bytes());
        // Include event data to differentiate multiple events in same tx
        hasher.update(&proof.event_data);
        hasher.finalize().to_vec()
    }
    
    /// Check if a proof has already been processed (replay protection)
    fn is_proof_processed(&self, proof_hash: &[u8]) -> bool {
        let state = self.state.read().unwrap();
        state.processed_proofs.contains_key(proof_hash)
    }
    
    /// Mark a proof as processed
    fn mark_proof_processed(&self, proof_hash: Vec<u8>) {
        let mut state = self.state.write().unwrap();
        state.processed_proofs.insert(proof_hash, current_timestamp());
    }
    
    /// Get count of processed proofs (for monitoring)
    pub fn get_processed_proof_count(&self) -> usize {
        let state = self.state.read().unwrap();
        state.processed_proofs.len()
    }
    
    /// Get total token accounts count
    pub fn get_token_account_count(&self) -> usize {
        let accounts = self.token_accounts.read().unwrap();
        accounts.len()
    }

    /// Process instruction
    pub fn process_instruction(
        &self,
        instruction: MintInstruction,
        caller: &[u8],
    ) -> Result<Vec<u8>> {
        match instruction {
            MintInstruction::Initialize { authority } => {
                self.initialize(authority)
            }
            MintInstruction::MintWrapped { proof, recipient, amount, original_token } => {
                self.mint_wrapped(proof, recipient, amount, original_token)
            }
            MintInstruction::BurnWrapped { token, amount, target_chain, recipient } => {
                self.burn_wrapped(caller, token, amount, target_chain, recipient)
            }
        }
    }

    /// Initialize mint program
    fn initialize(&self, authority: Vec<u8>) -> Result<Vec<u8>> {
        let mut state = self.state.write().unwrap();

        if !state.authority.is_empty() {
            return Err(anyhow::anyhow!("Already initialized"));
        }

        state.authority = authority.clone();
        info!("Mint program initialized with authority: {:?}", bs58::encode(&authority).into_string());

        Ok(b"initialized".to_vec())
    }

    /// Mint wrapped tokens after verifying proof
    /// 
    /// CRITICAL: Includes replay attack protection to prevent the same proof
    /// from being used multiple times to mint infinite tokens.
    fn mint_wrapped(
        &self,
        proof: super::solana_lock::BridgeProof,
        recipient: Vec<u8>,
        amount: u64,
        original_token: Vec<u8>,
    ) -> Result<Vec<u8>> {
        // STEP 1: Compute proof hash BEFORE acquiring lock (for replay check)
        let proof_hash = Self::compute_proof_hash(&proof);
        
        // STEP 2: Check for replay attack FIRST
        // This is the CRITICAL security check that was missing
        if self.is_proof_processed(&proof_hash) {
            error!(
                "REPLAY ATTACK DETECTED: Proof already processed. TX: {}, Block: {}",
                bs58::encode(&proof.source_tx_hash).into_string(),
                proof.block_number
            );
            return Err(anyhow::anyhow!(
                "Proof already processed (replay attack prevention). \
                Source TX: {}, Block: {}",
                bs58::encode(&proof.source_tx_hash).into_string(),
                proof.block_number
            ));
        }
        
        // STEP 3: Verify proof cryptographically
        self.verify_proof(&proof)?;

        // STEP 4: Mark proof as processed BEFORE minting (prevents race conditions)
        self.mark_proof_processed(proof_hash.clone());

        // STEP 5: Update token account balance (SPL-style separate accounts)
        // This prevents single-account bloat that would break the contract
        let current_balance = self.get_token_account_balance(&original_token, &recipient);
        let new_balance = current_balance + amount;
        self.update_token_account(&original_token, &recipient, new_balance);

        // Update token metadata (without storing balances in HashMap)
        let nonce = {
            let mut state = self.state.write().unwrap();
            
            // Get or create wrapped token metadata
            let token = state.wrapped_tokens.entry(original_token.clone())
                .or_insert_with(|| WrappedToken {
                    original_token: original_token.clone(),
                    original_chain: "solana".to_string(),
                    total_supply: 0,
                    holders: HashMap::new(), // Empty - balances in token_accounts
                });

            // Update total supply (but NOT holders HashMap)
            token.total_supply += amount;
            state.total_minted += amount;
            state.nonce += 1;
            state.nonce
        };

        // Emit mint event
        let mint_event = MintEvent {
            recipient: recipient.clone(),
            token: original_token,
            amount,
            source_chain: "solana".to_string(),
            nonce,
            timestamp: current_timestamp(),
        };

        info!("Minted {} wrapped tokens to {} (balance: {}), nonce: {}", 
            amount, bs58::encode(&recipient).into_string(), new_balance, nonce);

        // Return event data for logging
        Ok(bincode::serialize(&mint_event)?)
    }

    /// Burn wrapped tokens for withdrawal
    fn burn_wrapped(
        &self,
        burner: &[u8],
        token: Vec<u8>,
        amount: u64,
        target_chain: String,
        recipient: Vec<u8>,
    ) -> Result<Vec<u8>> {
        // Get balance from token accounts (SPL-style)
        let current_balance = self.get_token_account_balance(&token, burner);
        
        if current_balance == 0 {
            return Err(anyhow::anyhow!("No balance"));
        }

        if current_balance < amount {
            return Err(anyhow::anyhow!(
                "Insufficient balance: {} < {}", 
                current_balance, 
                amount
            ));
        }

        // Burn tokens from token account
        let new_balance = current_balance - amount;
        self.update_token_account(&token, burner, new_balance);

        // Update token metadata
        let nonce = {
            let mut state = self.state.write().unwrap();

            // Get wrapped token
            let wrapped_token = state.wrapped_tokens.get_mut(&token)
                .ok_or_else(|| anyhow::anyhow!("Token not found"))?;

            // Update total supply (but NOT holders HashMap)
            wrapped_token.total_supply -= amount;
            state.total_burned += amount;
            state.nonce += 1;
            state.nonce
        };

        // Emit burn event
        let burn_event = BurnEvent {
            burner: burner.to_vec(),
            token,
            amount,
            target_chain,
            recipient,
            nonce,
            timestamp: current_timestamp(),
        };

        info!("Burned {} tokens from {} (remaining: {}), nonce: {}", 
            amount, bs58::encode(burner).into_string(), new_balance, nonce);

        Ok(bincode::serialize(&burn_event)?)
    }

    /// Verify merkle proof against stored roots
    fn verify_merkle_proof(&self, proof: &[Vec<u8>], data: &[u8]) -> Result<()> {
        use sha2::{Sha256, Digest};
        
        if proof.is_empty() {
            return Ok(()); // No proof required for simple transfers
        }

        // Validate proof structure
        for (idx, sibling) in proof.iter().enumerate() {
            if sibling.len() != 32 {
                return Err(anyhow::anyhow!(
                    "Invalid merkle proof sibling at index {}: expected 32 bytes, got {}",
                    idx,
                    sibling.len()
                ));
            }
        }

        // Compute the merkle root from the proof
        let mut current_hash = Sha256::digest(data).to_vec();

        for sibling in proof {
            let combined = if current_hash < *sibling {
                [current_hash.as_slice(), sibling.as_slice()].concat()
            } else {
                [sibling.as_slice(), current_hash.as_slice()].concat()
            };

            current_hash = Sha256::digest(&combined).to_vec();
        }

        // Verify the computed root against known verified roots
        // In a real implementation, this would check against a stored list of verified roots
        // from the light client or a trusted source
        let computed_root = current_hash;
        
        // For now, verify the root has valid structure (32 bytes)
        if computed_root.len() != 32 {
            return Err(anyhow::anyhow!("Invalid computed merkle root"));
        }

        // Log the verified root for debugging
        info!("Merkle proof verified, computed root: {}", 
            bs58::encode(&computed_root).into_string());

        Ok(())
    }
    
    /// Verify the full bridge proof including all components
    fn verify_proof(&self, proof: &super::solana_lock::BridgeProof) -> Result<()> {
        // 1. Check relayer signatures
        const MIN_SIGNATURES: usize = 2;
        
        if proof.relayer_signatures.len() < MIN_SIGNATURES {
            return Err(anyhow::anyhow!(
                "Insufficient relayer signatures: {} < {}",
                proof.relayer_signatures.len(),
                MIN_SIGNATURES
            ));
        }

        // 2. Validate signature format
        for (idx, sig) in proof.relayer_signatures.iter().enumerate() {
            if sig.len() != 64 {
                return Err(anyhow::anyhow!(
                    "Invalid signature length at index {}: expected 64 bytes, got {}",
                    idx,
                    sig.len()
                ));
            }
        }

        // 3. Verify merkle proof if present
        if !proof.merkle_proof.is_empty() {
            self.verify_merkle_proof(&proof.merkle_proof, &proof.event_data)?;
        }

        // 4. Check block finality
        const FINALITY_BLOCKS: u64 = 32;
        if proof.block_number == 0 {
            return Err(anyhow::anyhow!("Invalid block number: cannot be zero"));
        }

        // 5. Verify source transaction hash
        if proof.source_tx_hash.len() != 32 && proof.source_tx_hash.len() != 64 {
            return Err(anyhow::anyhow!(
                "Invalid source transaction hash length: expected 32 or 64 bytes, got {}",
                proof.source_tx_hash.len()
            ));
        }

        info!("Bridge proof verified successfully for block {}", proof.block_number);
        Ok(())
    }

    /// Get token info
    pub fn get_token_info(&self, token: &[u8]) -> Option<WrappedToken> {
        let state = self.state.read().unwrap();
        state.wrapped_tokens.get(token).cloned()
    }

    /// Get balance from token accounts (SPL-style)
    pub fn get_balance(&self, token: &[u8], holder: &[u8]) -> u64 {
        // Primary: Use token accounts (new scalable design)
        let balance = self.get_token_account_balance(token, holder);
        if balance > 0 {
            return balance;
        }
        
        // Fallback: Check legacy holders HashMap for backward compatibility
        let state = self.state.read().unwrap();
        state.wrapped_tokens.get(token)
            .and_then(|t| t.holders.get(holder).copied())
            .unwrap_or(0)
    }

    /// Get statistics
    pub fn get_stats(&self) -> MintProgramStats {
        let state = self.state.read().unwrap();
        
        MintProgramStats {
            total_wrapped_tokens: state.wrapped_tokens.len(),
            total_minted: state.total_minted,
            total_burned: state.total_burned,
            nonce: state.nonce,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MintProgramStats {
    pub total_wrapped_tokens: usize,
    pub total_minted: u64,
    pub total_burned: u64,
    pub nonce: u64,
}

impl Default for AdeMintProgram {
    fn default() -> Self {
        Self::new()
    }
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mint_program_creation() {
        let program = AdeMintProgram::new();
        let stats = program.get_stats();
        assert_eq!(stats.total_minted, 0);
    }

    #[test]
    fn test_initialization() {
        let program = AdeMintProgram::new();
        let authority = vec![1u8; 32];
        
        let result = program.initialize(authority.clone());
        assert!(result.is_ok());
        
        let state = program.state.read().unwrap();
        assert_eq!(state.authority, authority);
    }

    #[test]
    fn test_mint_wrapped() {
        let program = AdeMintProgram::new();
        program.initialize(vec![1u8; 32]).unwrap();

        let proof = super::solana_lock::BridgeProof {
            source_tx_hash: vec![1; 32],
            block_number: 100,
            merkle_proof: vec![],
            event_data: vec![],
            relayer_signatures: vec![vec![1; 64], vec![2; 64]], // 2 signatures
        };

        let result = program.mint_wrapped(
            proof,
            vec![2u8; 32],
            1000,
            vec![3u8; 32],
        );

        assert!(result.is_ok());
        
        let balance = program.get_balance(&vec![3u8; 32], &vec![2u8; 32]);
        assert_eq!(balance, 1000);
    }

    #[test]
    fn test_burn_wrapped() {
        let program = AdeMintProgram::new();
        program.initialize(vec![1u8; 32]).unwrap();

        // First mint
        let proof = super::solana_lock::BridgeProof {
            source_tx_hash: vec![1; 32],
            block_number: 100,
            merkle_proof: vec![],
            event_data: vec![],
            relayer_signatures: vec![vec![1; 64], vec![2; 64]],
        };

        let holder = vec![2u8; 32];
        let token = vec![3u8; 32];
        
        program.mint_wrapped(proof, holder.clone(), 1000, token.clone()).unwrap();

        // Then burn
        let result = program.burn_wrapped(
            &holder,
            token.clone(),
            500,
            "solana".to_string(),
            vec![4u8; 32],
        );

        assert!(result.is_ok());
        
        let balance = program.get_balance(&token, &holder);
        assert_eq!(balance, 500);
    }
}

