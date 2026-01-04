use anyhow::{Result, Context};
use ed25519_dalek::{PublicKey, Signature, Verifier};
use sha2::{Sha256, Digest};
use serde::{Serialize, Deserialize};

use crate::merkle::MerkleTree;

/// Proof verification engine
pub struct ProofVerifier {
    authorized_relayers: Vec<Vec<u8>>,
    signature_threshold: usize,
    finality_threshold: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub valid: bool,
    pub relayers_verified: usize,
    pub merkle_verified: bool,
    pub finality_verified: bool,
    pub errors: Vec<String>,
}

impl ProofVerifier {
    pub fn new(
        authorized_relayers: Vec<Vec<u8>>,
        signature_threshold: usize,
        finality_threshold: u64,
    ) -> Self {
        Self {
            authorized_relayers,
            signature_threshold,
            finality_threshold,
        }
    }

    /// Comprehensive proof verification
    pub fn verify_proof(
        &self,
        proof: &crate::solana_lock::BridgeProof,
        current_block: u64,
    ) -> Result<VerificationResult> {
        let mut errors = Vec::new();
        let mut relayers_verified = 0;
        let mut merkle_verified = false;
        let mut finality_verified = false;

        // 1. Verify relayer signatures
        let message = self.construct_proof_message(proof);
        
        for (idx, sig_bytes) in proof.relayer_signatures.iter().enumerate() {
            if sig_bytes.len() != 64 {
                errors.push(format!("Invalid signature length at index {}", idx));
                continue;
            }

            // In production, verify against authorized relayer pubkeys
            if idx < self.authorized_relayers.len() {
                match self.verify_relayer_signature(&message, sig_bytes, &self.authorized_relayers[idx]) {
                    Ok(true) => {
                        relayers_verified += 1;
                    }
                    Ok(false) => {
                        errors.push(format!("Signature verification failed at index {}", idx));
                    }
                    Err(e) => {
                        errors.push(format!("Signature error at index {}: {}", idx, e));
                    }
                }
            }
        }

        // Check signature threshold
        if relayers_verified < self.signature_threshold {
            errors.push(format!(
                "Insufficient valid signatures: {} < {}",
                relayers_verified,
                self.signature_threshold
            ));
        }

        // 2. Verify merkle proof
        if !proof.merkle_proof.is_empty() {
            match self.verify_merkle_path(&proof.merkle_proof, &proof.event_data) {
                Ok(root) => {
                    merkle_verified = true;
                    // In production, verify root against known block root
                }
                Err(e) => {
                    errors.push(format!("Merkle proof verification failed: {}", e));
                }
            }
        } else {
            merkle_verified = true; // No merkle proof required
        }

        // 3. Verify block finality
        if current_block >= proof.block_number + self.finality_threshold {
            finality_verified = true;
        } else {
            errors.push(format!(
                "Block not finalized: current {} < {} + {}",
                current_block,
                proof.block_number,
                self.finality_threshold
            ));
        }

        let valid = relayers_verified >= self.signature_threshold
            && merkle_verified
            && finality_verified
            && errors.is_empty();

        Ok(VerificationResult {
            valid,
            relayers_verified,
            merkle_verified,
            finality_verified,
            errors,
        })
    }

    /// Construct message for signature verification
    fn construct_proof_message(&self, proof: &crate::solana_lock::BridgeProof) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(&proof.source_tx_hash);
        hasher.update(&proof.block_number.to_le_bytes());
        hasher.update(&proof.event_data);
        hasher.finalize().to_vec()
    }

    /// Verify relayer signature
    fn verify_relayer_signature(
        &self,
        message: &[u8],
        signature: &[u8],
        relayer_pubkey: &[u8],
    ) -> Result<bool> {
        if relayer_pubkey.len() != 32 {
            return Err(anyhow::anyhow!("Invalid pubkey length"));
        }

        if signature.len() != 64 {
            return Err(anyhow::anyhow!("Invalid signature length"));
        }

        let pubkey = PublicKey::from_bytes(relayer_pubkey)
            .context("Invalid relayer public key")?;

        let sig = Signature::from_bytes(signature)
            .context("Invalid signature format")?;

        match pubkey.verify(message, &sig) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Verify merkle proof path
    fn verify_merkle_path(&self, proof: &[Vec<u8>], leaf: &[u8]) -> Result<Vec<u8>> {
        let mut current_hash = Sha256::digest(leaf).to_vec();

        for (idx, sibling) in proof.iter().enumerate() {
            if sibling.len() != 32 {
                return Err(anyhow::anyhow!("Invalid sibling hash at index {}", idx));
            }

            let combined = if current_hash < *sibling {
                [current_hash.as_slice(), sibling.as_slice()].concat()
            } else {
                [sibling.as_slice(), current_hash.as_slice()].concat()
            };

            current_hash = Sha256::digest(&combined).to_vec();
        }

        Ok(current_hash)
    }

    /// Verify event data integrity
    pub fn verify_event_data(event_data: &[u8]) -> Result<bool> {
        // Parse event data
        if event_data.is_empty() {
            return Err(anyhow::anyhow!("Empty event data"));
        }

        // In production, deserialize and validate structure
        Ok(true)
    }
}

/// Proof builder for relayers
pub struct ProofBuilder {
    source_tx_hash: Vec<u8>,
    block_number: u64,
    event_data: Vec<u8>,
    merkle_proof: Vec<Vec<u8>>,
    relayer_signatures: Vec<Vec<u8>>,
}

impl ProofBuilder {
    pub fn new(source_tx_hash: Vec<u8>, block_number: u64, event_data: Vec<u8>) -> Self {
        Self {
            source_tx_hash,
            block_number,
            event_data,
            merkle_proof: Vec::new(),
            relayer_signatures: Vec::new(),
        }
    }

    pub fn add_merkle_proof(mut self, merkle_proof: Vec<Vec<u8>>) -> Self {
        self.merkle_proof = merkle_proof;
        self
    }

    pub fn add_signature(mut self, signature: Vec<u8>) -> Self {
        self.relayer_signatures.push(signature);
        self
    }

    pub fn build(self) -> crate::solana_lock::BridgeProof {
        crate::solana_lock::BridgeProof {
            source_tx_hash: self.source_tx_hash,
            block_number: self.block_number,
            merkle_proof: self.merkle_proof,
            event_data: self.event_data,
            relayer_signatures: self.relayer_signatures,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::Keypair;
    use ed25519_dalek::Signer;
    use rand::rngs::OsRng;

    #[test]
    fn test_proof_verifier() {
        let mut csprng = OsRng;
        let relayer1 = Keypair::generate(&mut csprng);
        let relayer2 = Keypair::generate(&mut csprng);

        let relayers = vec![
            relayer1.public.to_bytes().to_vec(),
            relayer2.public.to_bytes().to_vec(),
        ];

        let verifier = ProofVerifier::new(relayers, 2, 32);

        let proof = crate::solana_lock::BridgeProof {
            source_tx_hash: vec![1; 32],
            block_number: 100,
            merkle_proof: vec![],
            event_data: vec![1, 2, 3],
            relayer_signatures: vec![],
        };

        // Sign proof
        let message = verifier.construct_proof_message(&proof);
        let sig1 = relayer1.sign(&message);
        let sig2 = relayer2.sign(&message);

        let mut signed_proof = proof.clone();
        signed_proof.relayer_signatures = vec![
            sig1.to_bytes().to_vec(),
            sig2.to_bytes().to_vec(),
        ];

        let result = verifier.verify_proof(&signed_proof, 150).unwrap();
        
        assert!(result.valid);
        assert_eq!(result.relayers_verified, 2);
        assert!(result.finality_verified);
    }

    #[test]
    fn test_insufficient_signatures() {
        let verifier = ProofVerifier::new(vec![vec![1; 32]], 2, 32);

        let proof = crate::solana_lock::BridgeProof {
            source_tx_hash: vec![1; 32],
            block_number: 100,
            merkle_proof: vec![],
            event_data: vec![],
            relayer_signatures: vec![vec![1; 64]], // Only 1 signature
        };

        let result = verifier.verify_proof(&proof, 150).unwrap();
        
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.contains("Insufficient")));
    }

    #[test]
    fn test_proof_builder() {
        let proof = ProofBuilder::new(vec![1; 32], 100, vec![1, 2, 3])
            .add_merkle_proof(vec![vec![2; 32], vec![3; 32]])
            .add_signature(vec![4; 64])
            .add_signature(vec![5; 64])
            .build();

        assert_eq!(proof.block_number, 100);
        assert_eq!(proof.merkle_proof.len(), 2);
        assert_eq!(proof.relayer_signatures.len(), 2);
    }
}

