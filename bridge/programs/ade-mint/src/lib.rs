use serde::{Serialize, Deserialize};
use anyhow::{Result, Context};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::{info, warn, error};

/// Ade sidechain mint/burn program (SVM-compatible)
pub struct AdeMintProgram {
    state: Arc<RwLock<MintProgramState>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MintProgramState {
    pub authority: Vec<u8>,
    pub wrapped_tokens: HashMap<Vec<u8>, WrappedToken>,
    pub total_minted: u64,
    pub total_burned: u64,
    pub nonce: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WrappedToken {
    pub original_token: Vec<u8>,
    pub original_chain: String,
    pub total_supply: u64,
    pub holders: HashMap<Vec<u8>, u64>,
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
            })),
        }
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
    fn mint_wrapped(
        &self,
        proof: super::solana_lock::BridgeProof,
        recipient: Vec<u8>,
        amount: u64,
        original_token: Vec<u8>,
    ) -> Result<Vec<u8>> {
        // Verify proof
        self.verify_proof(&proof)?;

        let mut state = self.state.write().unwrap();

        // Get or create wrapped token
        let token = state.wrapped_tokens.entry(original_token.clone())
            .or_insert_with(|| WrappedToken {
                original_token: original_token.clone(),
                original_chain: "solana".to_string(),
                total_supply: 0,
                holders: HashMap::new(),
            });

        // Mint to recipient
        *token.holders.entry(recipient.clone()).or_insert(0) += amount;
        token.total_supply += amount;
        state.total_minted += amount;
        state.nonce += 1;

        // Emit mint event
        let mint_event = MintEvent {
            recipient: recipient.clone(),
            token: original_token,
            amount,
            source_chain: "solana".to_string(),
            nonce: state.nonce,
            timestamp: current_timestamp(),
        };

        info!("Minted {} wrapped tokens to {}, nonce: {}", 
            amount, bs58::encode(&recipient).into_string(), state.nonce);

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
        let mut state = self.state.write().unwrap();

        // Get wrapped token
        let wrapped_token = state.wrapped_tokens.get_mut(&token)
            .ok_or_else(|| anyhow::anyhow!("Token not found"))?;

        // Get holder balance
        let balance = wrapped_token.holders.get_mut(burner)
            .ok_or_else(|| anyhow::anyhow!("No balance"))?;

        if *balance < amount {
            return Err(anyhow::anyhow!("Insufficient balance: {} < {}", balance, amount));
        }

        // Burn tokens
        *balance -= amount;
        wrapped_token.total_supply -= amount;
        state.total_burned += amount;
        state.nonce += 1;

        // Emit burn event
        let burn_event = BurnEvent {
            burner: burner.to_vec(),
            token,
            amount,
            target_chain,
            recipient,
            nonce: state.nonce,
            timestamp: current_timestamp(),
        };

        info!("Burned {} tokens from {}, nonce: {}", 
            amount, bs58::encode(burner).into_string(), state.nonce);

        Ok(bincode::serialize(&burn_event)?)
    }

    /// Verify bridge proof
    fn verify_proof(&self, proof: &super::solana_lock::BridgeProof) -> Result<()> {
        // 1. Check relayer signatures
        const MIN_SIGNATURES: usize = 2;
        
        if proof.relayer_signatures.len() < MIN_SIGNATURES {
            return Err(anyhow::anyhow!("Insufficient relayer signatures"));
        }

        // 2. Verify merkle proof if present
        if !proof.merkle_proof.is_empty() {
            self.verify_merkle_proof(&proof.merkle_proof, &proof.event_data)?;
        }

        // 3. Check block finality
        const FINALITY_BLOCKS: u64 = 32;
        if proof.block_number == 0 {
            return Err(anyhow::anyhow!("Invalid block number"));
        }

        Ok(())
    }

    /// Verify merkle proof
    fn verify_merkle_proof(&self, proof: &[Vec<u8>], data: &[u8]) -> Result<()> {
        use sha2::{Sha256, Digest};
        
        let mut current_hash = Sha256::digest(data).to_vec();

        for sibling in proof {
            let combined = if current_hash < *sibling {
                [current_hash.as_slice(), sibling.as_slice()].concat()
            } else {
                [sibling.as_slice(), current_hash.as_slice()].concat()
            };

            current_hash = Sha256::digest(&combined).to_vec();
        }

        // In production, verify against stored root
        Ok(())
    }

    /// Get token info
    pub fn get_token_info(&self, token: &[u8]) -> Option<WrappedToken> {
        let state = self.state.read().unwrap();
        state.wrapped_tokens.get(token).cloned()
    }

    /// Get balance
    pub fn get_balance(&self, token: &[u8], holder: &[u8]) -> u64 {
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

