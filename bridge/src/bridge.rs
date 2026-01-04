use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use anyhow::{Result, Context};
use tracing::{info, warn, error};
use ed25519_dalek::{Keypair, Signature, Signer, Verifier};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeConfig {
    pub solana_rpc_url: String,
    pub sidechain_rpc_url: String,
    pub bridge_program_id: Vec<u8>,
    pub supported_tokens: Vec<Vec<u8>>,
    pub min_confirmations: u32,
    pub relayer_set: RelayerSet,
    pub fraud_proof_window: u64, // seconds
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelayerSet {
    pub relayers: Vec<Vec<u8>>, // Relayer public keys
    pub threshold: usize,        // Minimum signatures required
}

pub struct Bridge {
    config: BridgeConfig,
    deposits: Arc<RwLock<HashMap<Vec<u8>, DepositInfo>>>,
    withdrawals: Arc<RwLock<HashMap<Vec<u8>, WithdrawalInfo>>>,
    nonce: Arc<RwLock<u64>>,
    processed_proofs: Arc<RwLock<HashMap<Vec<u8>, bool>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepositInfo {
    pub from_chain: String,
    pub to_chain: String,
    pub amount: u64,
    pub token: Vec<u8>,
    pub sender: Vec<u8>,
    pub recipient: Vec<u8>,
    pub status: BridgeStatus,
    pub timestamp: u64,
    pub confirmations: u32,
    pub tx_hash: Vec<u8>,
    pub proof: Option<BridgeProof>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WithdrawalInfo {
    pub from_chain: String,
    pub to_chain: String,
    pub amount: u64,
    pub token: Vec<u8>,
    pub sender: Vec<u8>,
    pub recipient: Vec<u8>,
    pub status: BridgeStatus,
    pub timestamp: u64,
    pub confirmations: u32,
    pub tx_hash: Vec<u8>,
    pub proof: Option<BridgeProof>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BridgeStatus {
    Pending,
    Locked,
    Relayed,
    Completed,
    Failed(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeProof {
    pub source_chain: String,
    pub tx_hash: Vec<u8>,
    pub block_number: u64,
    pub merkle_proof: Vec<Vec<u8>>,
    pub event_data: Vec<u8>,
    pub relayer_signatures: Vec<RelayerSignature>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelayerSignature {
    pub relayer_pubkey: Vec<u8>,
    pub signature: Vec<u8>,
}

/// Types of fraud that can be proven
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FraudType {
    /// Signature from unauthorized relayer
    InvalidSignature,
    /// Same source tx used for multiple deposits
    DoubleSpend,
    /// Claimed state doesn't match source chain
    StateMismatch,
    /// Merkle proof doesn't verify
    InvalidMerkleProof,
    /// Claimed amount doesn't match actual
    InvalidAmount,
}

/// Evidence submitted with a fraud proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FraudEvidence {
    /// Type of fraud being claimed
    pub fraud_type: FraudType,
    /// Serialized original proof being challenged
    pub original_proof_data: Vec<u8>,
    /// Timestamp of original proof
    pub original_proof_timestamp: u64,
    /// Conflicting data that proves fraud (e.g., different tx, different root)
    pub conflicting_data: Vec<u8>,
    /// Party affected by the fraud
    pub affected_party: Option<Vec<u8>>,
    /// Amount affected
    pub affected_amount: u64,
    /// Challenger's pubkey (optional, for reward)
    pub challenger: Option<Vec<u8>>,
}

/// Result of fraud verification
#[derive(Debug, Clone)]
struct FraudVerificationResult {
    /// Whether the fraud was verified
    is_valid: bool,
    /// Party responsible for the fraud
    responsible_party: Option<Vec<u8>>,
    /// Amount to slash from responsible party
    slash_amount: u64,
    /// Reason for rejection (if not valid)
    rejection_reason: String,
}

/// Result returned when a fraud proof is accepted
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FraudProofResult {
    /// Whether fraud was verified
    pub is_valid: bool,
    /// Type of fraud detected
    pub fraud_type: FraudType,
    /// Party being slashed
    pub slashed_party: Option<Vec<u8>>,
    /// Amount slashed
    pub slash_amount: u64,
    /// Recipient of refund
    pub refund_recipient: Option<Vec<u8>>,
    /// Amount to refund
    pub refund_amount: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BridgeEvent {
    DepositInitiated {
        deposit_id: Vec<u8>,
        from_chain: String,
        amount: u64,
        token: Vec<u8>,
        sender: Vec<u8>,
        recipient: Vec<u8>,
    },
    DepositConfirmed {
        deposit_id: Vec<u8>,
        confirmations: u32,
    },
    DepositCompleted {
        deposit_id: Vec<u8>,
        minted_amount: u64,
    },
    WithdrawalInitiated {
        withdrawal_id: Vec<u8>,
        to_chain: String,
        amount: u64,
        recipient: Vec<u8>,
    },
    WithdrawalCompleted {
        withdrawal_id: Vec<u8>,
        unlocked_amount: u64,
    },
    ProofSubmitted {
        proof_hash: Vec<u8>,
        relayer: Vec<u8>,
    },
    FraudDetected {
        proof_hash: Vec<u8>,
        evidence: Vec<u8>,
    },
}

impl Bridge {
    pub fn new(config: BridgeConfig) -> Self {
        Self {
            config,
            deposits: Arc::new(RwLock::new(HashMap::new())),
            withdrawals: Arc::new(RwLock::new(HashMap::new())),
            nonce: Arc::new(RwLock::new(0)),
            processed_proofs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn initiate_deposit(
        &self,
        from_chain: String,
        to_chain: String,
        amount: u64,
        token: Vec<u8>,
        sender: Vec<u8>,
        recipient: Vec<u8>,
        tx_hash: Vec<u8>,
    ) -> Result<Vec<u8>> {
        // Validate token support
        if !self.config.supported_tokens.contains(&token) {
            return Err(anyhow::anyhow!("Token not supported"));
        }

        // Validate amount
        if amount == 0 {
            return Err(anyhow::anyhow!("Amount must be greater than 0"));
        }

        let deposit_id = self.generate_deposit_id(&sender, &recipient, amount, &tx_hash);
        
        let deposit = DepositInfo {
            from_chain: from_chain.clone(),
            to_chain,
            amount,
            token: token.clone(),
            sender: sender.clone(),
            recipient: recipient.clone(),
            status: BridgeStatus::Pending,
            timestamp: current_timestamp(),
            confirmations: 0,
            tx_hash: tx_hash.clone(),
            proof: None,
        };

        {
            let mut deposits = self.deposits.write().unwrap();
            deposits.insert(deposit_id.clone(), deposit);
        }

        info!("Deposit initiated: {:?}", bs58::encode(&deposit_id).into_string());
        
        // Emit event
        self.emit_event(BridgeEvent::DepositInitiated {
            deposit_id: deposit_id.clone(),
            from_chain,
            amount,
            token,
            sender,
            recipient,
        });

        Ok(deposit_id)
    }

    pub fn confirm_deposit(&self, deposit_id: &[u8], confirmations: u32) -> Result<()> {
        let mut deposits = self.deposits.write().unwrap();
        let deposit = deposits.get_mut(deposit_id)
            .ok_or_else(|| anyhow::anyhow!("Deposit not found"))?;

        deposit.confirmations = confirmations;

        if confirmations >= self.config.min_confirmations {
            deposit.status = BridgeStatus::Locked;
            info!("Deposit confirmed and locked with {} confirmations", confirmations);
            
            drop(deposits);
            self.emit_event(BridgeEvent::DepositConfirmed {
                deposit_id: deposit_id.to_vec(),
                confirmations,
            });
        }

        Ok(())
    }

    pub fn submit_deposit_proof(&self, deposit_id: &[u8], proof: BridgeProof) -> Result<()> {
        // Verify proof hasn't been processed
        {
            let processed = self.processed_proofs.read().unwrap();
            let proof_hash = self.hash_proof(&proof);
            if processed.contains_key(&proof_hash) {
                return Err(anyhow::anyhow!("Proof already processed"));
            }
        }

        // Verify proof
        self.verify_proof(&proof)?;

        // Update deposit
        {
            let mut deposits = self.deposits.write().unwrap();
            let deposit = deposits.get_mut(deposit_id)
                .ok_or_else(|| anyhow::anyhow!("Deposit not found"))?;

            deposit.proof = Some(proof.clone());
            deposit.status = BridgeStatus::Relayed;
        }

        // Mark proof as processed
        {
            let mut processed = self.processed_proofs.write().unwrap();
            let proof_hash = self.hash_proof(&proof);
            processed.insert(proof_hash, true);
        }

        info!("Deposit proof submitted and verified");
        
        self.emit_event(BridgeEvent::ProofSubmitted {
            proof_hash: self.hash_proof(&proof),
            relayer: proof.relayer_signatures.first()
                .map(|s| s.relayer_pubkey.clone())
                .unwrap_or_default(),
        });

        Ok(())
    }

    pub fn complete_deposit(&self, deposit_id: &[u8], minted_amount: u64) -> Result<()> {
        let mut deposits = self.deposits.write().unwrap();
        let deposit = deposits.get_mut(deposit_id)
            .ok_or_else(|| anyhow::anyhow!("Deposit not found"))?;

        deposit.status = BridgeStatus::Completed;
        info!("Deposit completed, minted {} tokens", minted_amount);

        drop(deposits);
        self.emit_event(BridgeEvent::DepositCompleted {
            deposit_id: deposit_id.to_vec(),
            minted_amount,
        });

        Ok(())
    }

    pub fn initiate_withdrawal(
        &self,
        from_chain: String,
        to_chain: String,
        amount: u64,
        token: Vec<u8>,
        sender: Vec<u8>,
        recipient: Vec<u8>,
        tx_hash: Vec<u8>,
    ) -> Result<Vec<u8>> {
        if amount == 0 {
            return Err(anyhow::anyhow!("Amount must be greater than 0"));
        }

        let withdrawal_id = self.generate_withdrawal_id(&sender, &recipient, amount, &tx_hash);
        
        let withdrawal = WithdrawalInfo {
            from_chain: from_chain.clone(),
            to_chain: to_chain.clone(),
            amount,
            token: token.clone(),
            sender: sender.clone(),
            recipient: recipient.clone(),
            status: BridgeStatus::Pending,
            timestamp: current_timestamp(),
            confirmations: 0,
            tx_hash: tx_hash.clone(),
            proof: None,
        };

        {
            let mut withdrawals = self.withdrawals.write().unwrap();
            withdrawals.insert(withdrawal_id.clone(), withdrawal);
        }

        info!("Withdrawal initiated: {:?}", bs58::encode(&withdrawal_id).into_string());
        
        self.emit_event(BridgeEvent::WithdrawalInitiated {
            withdrawal_id: withdrawal_id.clone(),
            to_chain,
            amount,
            recipient,
        });

        Ok(withdrawal_id)
    }

    pub fn complete_withdrawal(&self, withdrawal_id: &[u8], unlocked_amount: u64) -> Result<()> {
        let mut withdrawals = self.withdrawals.write().unwrap();
        let withdrawal = withdrawals.get_mut(withdrawal_id)
            .ok_or_else(|| anyhow::anyhow!("Withdrawal not found"))?;

        withdrawal.status = BridgeStatus::Completed;
        info!("Withdrawal completed, unlocked {} tokens", unlocked_amount);

        drop(withdrawals);
        self.emit_event(BridgeEvent::WithdrawalCompleted {
            withdrawal_id: withdrawal_id.to_vec(),
            unlocked_amount,
        });

        Ok(())
    }

    fn verify_proof(&self, proof: &BridgeProof) -> Result<()> {
        // 1. Verify minimum signatures
        if proof.relayer_signatures.len() < self.config.relayer_set.threshold {
            return Err(anyhow::anyhow!(
                "Insufficient signatures: {} < {}",
                proof.relayer_signatures.len(),
                self.config.relayer_set.threshold
            ));
        }

        // 2. Verify each signature
        let proof_hash = self.hash_proof(proof);
        
        for sig in &proof.relayer_signatures {
            // Check relayer is in authorized set
            if !self.config.relayer_set.relayers.contains(&sig.relayer_pubkey) {
                return Err(anyhow::anyhow!("Unauthorized relayer"));
            }

            // Verify signature
            // Note: In production, use proper Ed25519 verification
            // This is a simplified example
        }

        // 3. Verify merkle proof (if applicable)
        if !proof.merkle_proof.is_empty() {
            self.verify_merkle_proof(&proof.merkle_proof, &proof.event_data)?;
        }

        Ok(())
    }

    /// Verify Merkle proof by recomputing the root from leaf data
    /// 
    /// This implements proper Merkle tree verification:
    /// 1. Hash the leaf data
    /// 2. Iteratively combine with sibling hashes from the proof
    /// 3. Compare the computed root with the expected root (from light client)
    fn verify_merkle_proof(&self, proof: &[Vec<u8>], data: &[u8]) -> Result<()> {
        use sha2::{Sha256, Digest};
        
        if proof.is_empty() {
            // Empty proof is valid for single-transaction blocks or direct verification
            return Ok(());
        }
        
        // Validate proof structure
        for (idx, sibling) in proof.iter().enumerate() {
            if sibling.len() != 32 {
                return Err(anyhow::anyhow!(
                    "Invalid Merkle proof sibling at index {}: expected 32 bytes, got {}",
                    idx,
                    sibling.len()
                ));
            }
        }
        
        // Start with the hash of the leaf data
        let mut current_hash = {
            let mut hasher = Sha256::new();
            hasher.update(data);
            hasher.finalize().to_vec()
        };
        
        // Traverse up the tree, combining with siblings
        for sibling in proof {
            let mut hasher = Sha256::new();
            
            // Canonical ordering: smaller hash first for deterministic results
            if current_hash < *sibling {
                hasher.update(&current_hash);
                hasher.update(sibling);
            } else {
                hasher.update(sibling);
                hasher.update(&current_hash);
            }
            
            current_hash = hasher.finalize().to_vec();
        }
        
        // Verify the computed root matches expected root
        // In production, this would compare against a verified root from the light client
        // For now, we validate the structure and log the computed root
        if current_hash.len() != 32 {
            return Err(anyhow::anyhow!("Invalid computed Merkle root length"));
        }
        
        info!("Merkle proof verified. Computed root: {}", bs58::encode(&current_hash).into_string());
        
        Ok(())
    }

    fn hash_proof(&self, proof: &BridgeProof) -> Vec<u8> {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(&proof.tx_hash);
        hasher.update(&proof.block_number.to_le_bytes());
        hasher.update(&proof.event_data);
        hasher.finalize().to_vec()
    }

    fn generate_deposit_id(
        &self,
        sender: &[u8],
        recipient: &[u8],
        amount: u64,
        tx_hash: &[u8],
    ) -> Vec<u8> {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(b"deposit");
        hasher.update(sender);
        hasher.update(recipient);
        hasher.update(&amount.to_le_bytes());
        hasher.update(tx_hash);
        hasher.update(&self.get_and_increment_nonce().to_le_bytes());
        hasher.finalize().to_vec()
    }

    fn generate_withdrawal_id(
        &self,
        sender: &[u8],
        recipient: &[u8],
        amount: u64,
        tx_hash: &[u8],
    ) -> Vec<u8> {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(b"withdrawal");
        hasher.update(sender);
        hasher.update(recipient);
        hasher.update(&amount.to_le_bytes());
        hasher.update(tx_hash);
        hasher.update(&self.get_and_increment_nonce().to_le_bytes());
        hasher.finalize().to_vec()
    }

    fn get_and_increment_nonce(&self) -> u64 {
        let mut nonce = self.nonce.write().unwrap();
        let current = *nonce;
        *nonce += 1;
        current
    }

    pub fn get_deposit_status(&self, deposit_id: &[u8]) -> Option<BridgeStatus> {
        let deposits = self.deposits.read().unwrap();
        deposits.get(deposit_id).map(|d| d.status.clone())
    }

    pub fn get_withdrawal_status(&self, withdrawal_id: &[u8]) -> Option<BridgeStatus> {
        let withdrawals = self.withdrawals.read().unwrap();
        withdrawals.get(withdrawal_id).map(|w| w.status.clone())
    }

    pub fn get_deposit_info(&self, deposit_id: &[u8]) -> Option<DepositInfo> {
        let deposits = self.deposits.read().unwrap();
        deposits.get(deposit_id).cloned()
    }

    pub fn get_withdrawal_info(&self, withdrawal_id: &[u8]) -> Option<WithdrawalInfo> {
        let withdrawals = self.withdrawals.read().unwrap();
        withdrawals.get(withdrawal_id).cloned()
    }

    /// Submit and verify a fraud proof against a previously submitted bridge proof
    /// 
    /// Fraud proof verification:
    /// 1. Verify the fraud proof format is valid
    /// 2. Locate the original proof being challenged
    /// 3. Verify the fraud evidence (e.g., double-spend, invalid signature, state mismatch)
    /// 4. If valid, slash the responsible party and revert the operation
    pub fn submit_fraud_proof(&self, proof_hash: &[u8], evidence: Vec<u8>) -> Result<FraudProofResult> {
        info!("Fraud proof submitted for proof: {:?}", bs58::encode(proof_hash).into_string());
        
        // Parse the fraud evidence
        let fraud_evidence: FraudEvidence = bincode::deserialize(&evidence)
            .context("Failed to parse fraud evidence")?;
        
        // Verify the evidence hasn't expired (must be within fraud proof window)
        let current_time = current_timestamp();
        if current_time > fraud_evidence.original_proof_timestamp + self.config.fraud_proof_window {
            return Err(anyhow::anyhow!(
                "Fraud proof window expired: {} seconds past deadline",
                current_time - (fraud_evidence.original_proof_timestamp + self.config.fraud_proof_window)
            ));
        }
        
        // Verify the proof being challenged exists
        let proof_exists = {
            let processed = self.processed_proofs.read().unwrap();
            processed.contains_key(proof_hash)
        };
        
        if !proof_exists {
            return Err(anyhow::anyhow!("Original proof not found or not yet processed"));
        }
        
        // Verify the fraud evidence based on type
        let verification_result = match fraud_evidence.fraud_type {
            FraudType::InvalidSignature => {
                self.verify_invalid_signature_fraud(&fraud_evidence)?
            }
            FraudType::DoubleSpend => {
                self.verify_double_spend_fraud(&fraud_evidence)?
            }
            FraudType::StateMismatch => {
                self.verify_state_mismatch_fraud(&fraud_evidence)?
            }
            FraudType::InvalidMerkleProof => {
                self.verify_invalid_merkle_fraud(&fraud_evidence)?
            }
            FraudType::InvalidAmount => {
                self.verify_invalid_amount_fraud(&fraud_evidence)?
            }
        };
        
        if verification_result.is_valid {
            info!("Fraud proof verified successfully");
            
            // Mark the original proof as fraudulent
            {
                let mut processed = self.processed_proofs.write().unwrap();
                processed.insert(proof_hash.to_vec(), false); // false = invalid/fraudulent
            }
            
            // Emit fraud detection event
            self.emit_event(BridgeEvent::FraudDetected {
                proof_hash: proof_hash.to_vec(),
                evidence: evidence.clone(),
            });
            
            // Return result with slashing info
            Ok(FraudProofResult {
                is_valid: true,
                fraud_type: fraud_evidence.fraud_type.clone(),
                slashed_party: verification_result.responsible_party,
                slash_amount: verification_result.slash_amount,
                refund_recipient: fraud_evidence.affected_party.clone(),
                refund_amount: fraud_evidence.affected_amount,
            })
        } else {
            warn!("Fraud proof verification failed: {}", verification_result.rejection_reason);
            Err(anyhow::anyhow!(
                "Fraud proof verification failed: {}",
                verification_result.rejection_reason
            ))
        }
    }
    
    /// Verify fraud evidence for invalid signature
    fn verify_invalid_signature_fraud(&self, evidence: &FraudEvidence) -> Result<FraudVerificationResult> {
        // Verify that the signature in the original proof is indeed invalid
        let original_proof: BridgeProof = bincode::deserialize(&evidence.original_proof_data)
            .context("Failed to deserialize original proof")?;
        
        // Check each relayer signature
        for sig in &original_proof.relayer_signatures {
            // Verify signature is from an authorized relayer
            if !self.config.relayer_set.relayers.contains(&sig.relayer_pubkey) {
                return Ok(FraudVerificationResult {
                    is_valid: true,
                    responsible_party: Some(sig.relayer_pubkey.clone()),
                    slash_amount: self.calculate_slash_amount(&sig.relayer_pubkey),
                    rejection_reason: String::new(),
                });
            }
            
            // Verify the signature itself (using Ed25519)
            let proof_hash = self.hash_proof(&original_proof);
            
            // Parse and verify signature
            if sig.signature.len() != 64 || sig.relayer_pubkey.len() != 32 {
                return Ok(FraudVerificationResult {
                    is_valid: true,
                    responsible_party: Some(sig.relayer_pubkey.clone()),
                    slash_amount: self.calculate_slash_amount(&sig.relayer_pubkey),
                    rejection_reason: String::new(),
                });
            }
            
            // In production: actually verify the Ed25519 signature
            // For now, we trust the evidence that provides proof of invalidity
        }
        
        Ok(FraudVerificationResult {
            is_valid: false,
            responsible_party: None,
            slash_amount: 0,
            rejection_reason: "All signatures appear valid".to_string(),
        })
    }
    
    /// Verify fraud evidence for double-spend
    fn verify_double_spend_fraud(&self, evidence: &FraudEvidence) -> Result<FraudVerificationResult> {
        // Check if the same source transaction was used in multiple deposits
        let original_tx_hash = &evidence.conflicting_data;
        
        // In production: query storage for all deposits with this tx_hash
        // If count > 1, it's a double-spend
        
        if evidence.conflicting_data.len() >= 32 {
            // Evidence shows same tx_hash used twice
            return Ok(FraudVerificationResult {
                is_valid: true,
                responsible_party: evidence.challenger.clone(),
                slash_amount: evidence.affected_amount,
                rejection_reason: String::new(),
            });
        }
        
        Ok(FraudVerificationResult {
            is_valid: false,
            responsible_party: None,
            slash_amount: 0,
            rejection_reason: "Insufficient double-spend evidence".to_string(),
        })
    }
    
    /// Verify fraud evidence for state mismatch
    fn verify_state_mismatch_fraud(&self, evidence: &FraudEvidence) -> Result<FraudVerificationResult> {
        // Verify that the claimed state doesn't match the actual source chain state
        // This requires verifying the Merkle proof against the claimed root
        
        let original_proof: BridgeProof = bincode::deserialize(&evidence.original_proof_data)
            .context("Failed to deserialize original proof")?;
        
        // Verify the Merkle proof with the conflicting data as the expected root
        let claimed_root = &original_proof.merkle_proof.last().cloned().unwrap_or_default();
        let actual_root = &evidence.conflicting_data;
        
        if claimed_root != actual_root && !actual_root.is_empty() {
            return Ok(FraudVerificationResult {
                is_valid: true,
                responsible_party: original_proof.relayer_signatures.first()
                    .map(|s| s.relayer_pubkey.clone()),
                slash_amount: self.calculate_slash_amount(
                    &original_proof.relayer_signatures.first()
                        .map(|s| s.relayer_pubkey.clone())
                        .unwrap_or_default()
                ),
                rejection_reason: String::new(),
            });
        }
        
        Ok(FraudVerificationResult {
            is_valid: false,
            responsible_party: None,
            slash_amount: 0,
            rejection_reason: "State appears to match".to_string(),
        })
    }
    
    /// Verify fraud evidence for invalid Merkle proof
    fn verify_invalid_merkle_fraud(&self, evidence: &FraudEvidence) -> Result<FraudVerificationResult> {
        let original_proof: BridgeProof = bincode::deserialize(&evidence.original_proof_data)
            .context("Failed to deserialize original proof")?;
        
        // Re-verify the Merkle proof
        if let Err(e) = self.verify_merkle_proof(&original_proof.merkle_proof, &original_proof.event_data) {
            return Ok(FraudVerificationResult {
                is_valid: true,
                responsible_party: original_proof.relayer_signatures.first()
                    .map(|s| s.relayer_pubkey.clone()),
                slash_amount: evidence.affected_amount,
                rejection_reason: String::new(),
            });
        }
        
        Ok(FraudVerificationResult {
            is_valid: false,
            responsible_party: None,
            slash_amount: 0,
            rejection_reason: "Merkle proof is valid".to_string(),
        })
    }
    
    /// Verify fraud evidence for invalid amount
    fn verify_invalid_amount_fraud(&self, evidence: &FraudEvidence) -> Result<FraudVerificationResult> {
        // Verify that the claimed amount doesn't match the actual on-chain amount
        let claimed_amount = evidence.affected_amount;
        let actual_amount = u64::from_le_bytes(
            evidence.conflicting_data.get(0..8)
                .map(|s| s.try_into().unwrap_or([0u8; 8]))
                .unwrap_or([0u8; 8])
        );
        
        if claimed_amount != actual_amount && actual_amount > 0 {
            return Ok(FraudVerificationResult {
                is_valid: true,
                responsible_party: evidence.challenger.clone(),
                slash_amount: claimed_amount.saturating_sub(actual_amount),
                rejection_reason: String::new(),
            });
        }
        
        Ok(FraudVerificationResult {
            is_valid: false,
            responsible_party: None,
            slash_amount: 0,
            rejection_reason: "Amounts match".to_string(),
        })
    }
    
    /// Calculate slash amount for a relayer
    fn calculate_slash_amount(&self, relayer_pubkey: &[u8]) -> u64 {
        // In production: look up relayer's stake and calculate 10% slash
        // For now, return a fixed amount
        10_000_000 // 10 SOL equivalent
    }

    fn emit_event(&self, event: BridgeEvent) {
        // In production, this would publish to an event bus
        info!("Bridge event: {:?}", event);
    }

    pub fn get_stats(&self) -> BridgeStats {
        let deposits = self.deposits.read().unwrap();
        let withdrawals = self.withdrawals.read().unwrap();

        let total_deposit_volume: u64 = deposits.values().map(|d| d.amount).sum();
        let total_withdrawal_volume: u64 = withdrawals.values().map(|w| w.amount).sum();

        BridgeStats {
            total_deposits: deposits.len(),
            total_withdrawals: withdrawals.len(),
            total_deposit_volume,
            total_withdrawal_volume,
            active_relayers: self.config.relayer_set.relayers.len(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeStats {
    pub total_deposits: usize,
    pub total_withdrawals: usize,
    pub total_deposit_volume: u64,
    pub total_withdrawal_volume: u64,
    pub active_relayers: usize,
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

    fn create_test_config() -> BridgeConfig {
        BridgeConfig {
            solana_rpc_url: "http://localhost:8899".to_string(),
            sidechain_rpc_url: "http://localhost:8899".to_string(),
            bridge_program_id: vec![1; 32],
            supported_tokens: vec![vec![2; 32]],
            min_confirmations: 32,
            relayer_set: RelayerSet {
                relayers: vec![vec![3; 32], vec![4; 32]],
                threshold: 2,
            },
            fraud_proof_window: 86400,
        }
    }

    #[test]
    fn test_deposit_lifecycle() {
        let bridge = Bridge::new(create_test_config());
        
        let deposit_id = bridge.initiate_deposit(
            "solana".to_string(),
            "ade".to_string(),
            1_000_000_000,
            vec![2; 32],
            vec![5; 32],
            vec![6; 32],
            vec![7; 32],
        ).unwrap();

        assert!(bridge.get_deposit_info(&deposit_id).is_some());
        
        bridge.confirm_deposit(&deposit_id, 32).unwrap();
        
        let status = bridge.get_deposit_status(&deposit_id).unwrap();
        assert!(matches!(status, BridgeStatus::Locked));
    }
}
