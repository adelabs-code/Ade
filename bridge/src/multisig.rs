use ed25519_dalek::{PublicKey, Signature, Verifier};
use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use std::collections::HashSet;

/// Multi-signature relayer system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSigRelayer {
    pub relayers: Vec<Vec<u8>>,      // Authorized relayer public keys
    pub threshold: usize,              // Minimum signatures required
    pub current_signatures: Vec<RelayerSignature>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelayerSignature {
    pub relayer_pubkey: Vec<u8>,
    pub signature: Vec<u8>,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub enum MultiSigError {
    InsufficientSignatures { have: usize, need: usize },
    UnauthorizedRelayer(Vec<u8>),
    DuplicateSignature(Vec<u8>),
    InvalidSignature { relayer: Vec<u8>, error: String },
    InvalidThreshold { threshold: usize, relayers: usize },
}

impl std::fmt::Display for MultiSigError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::InsufficientSignatures { have, need } => {
                write!(f, "Insufficient signatures: have {}, need {}", have, need)
            }
            Self::UnauthorizedRelayer(pubkey) => {
                write!(f, "Unauthorized relayer: {:?}", bs58::encode(pubkey).into_string())
            }
            Self::DuplicateSignature(pubkey) => {
                write!(f, "Duplicate signature from: {:?}", bs58::encode(pubkey).into_string())
            }
            Self::InvalidSignature { relayer, error } => {
                write!(f, "Invalid signature from {:?}: {}", bs58::encode(relayer).into_string(), error)
            }
            Self::InvalidThreshold { threshold, relayers } => {
                write!(f, "Invalid threshold: {} > {}", threshold, relayers)
            }
        }
    }
}

impl std::error::Error for MultiSigError {}

impl MultiSigRelayer {
    /// Create a new multi-sig relayer configuration
    pub fn new(relayers: Vec<Vec<u8>>, threshold: usize) -> Result<Self, MultiSigError> {
        if threshold == 0 || threshold > relayers.len() {
            return Err(MultiSigError::InvalidThreshold {
                threshold,
                relayers: relayers.len(),
            });
        }

        Ok(Self {
            relayers,
            threshold,
            current_signatures: Vec::new(),
        })
    }

    /// Add a signature
    pub fn add_signature(&mut self, signature: RelayerSignature) -> Result<(), MultiSigError> {
        // Check if relayer is authorized
        if !self.relayers.contains(&signature.relayer_pubkey) {
            return Err(MultiSigError::UnauthorizedRelayer(signature.relayer_pubkey.clone()));
        }

        // Check for duplicate
        for existing in &self.current_signatures {
            if existing.relayer_pubkey == signature.relayer_pubkey {
                return Err(MultiSigError::DuplicateSignature(signature.relayer_pubkey.clone()));
            }
        }

        self.current_signatures.push(signature);
        Ok(())
    }

    /// Verify all signatures against a message
    pub fn verify_signatures(&self, message: &[u8]) -> Result<bool, MultiSigError> {
        if self.current_signatures.len() < self.threshold {
            return Err(MultiSigError::InsufficientSignatures {
                have: self.current_signatures.len(),
                need: self.threshold,
            });
        }

        // Track unique signers
        let mut unique_signers = HashSet::new();

        for sig in &self.current_signatures {
            // Check authorization
            if !self.relayers.contains(&sig.relayer_pubkey) {
                return Err(MultiSigError::UnauthorizedRelayer(sig.relayer_pubkey.clone()));
            }

            // Check for duplicates
            if !unique_signers.insert(sig.relayer_pubkey.clone()) {
                return Err(MultiSigError::DuplicateSignature(sig.relayer_pubkey.clone()));
            }

            // Verify signature
            let pubkey = PublicKey::from_bytes(&sig.relayer_pubkey)
                .map_err(|e| MultiSigError::InvalidSignature {
                    relayer: sig.relayer_pubkey.clone(),
                    error: e.to_string(),
                })?;

            let signature = Signature::from_bytes(&sig.signature)
                .map_err(|e| MultiSigError::InvalidSignature {
                    relayer: sig.relayer_pubkey.clone(),
                    error: e.to_string(),
                })?;

            pubkey.verify(message, &signature)
                .map_err(|e| MultiSigError::InvalidSignature {
                    relayer: sig.relayer_pubkey.clone(),
                    error: e.to_string(),
                })?;
        }

        // Check threshold
        if unique_signers.len() < self.threshold {
            return Err(MultiSigError::InsufficientSignatures {
                have: unique_signers.len(),
                need: self.threshold,
            });
        }

        Ok(true)
    }

    /// Check if threshold is met
    pub fn has_threshold(&self) -> bool {
        self.current_signatures.len() >= self.threshold
    }

    /// Get signature count
    pub fn signature_count(&self) -> usize {
        self.current_signatures.len()
    }

    /// Clear signatures
    pub fn clear_signatures(&mut self) {
        self.current_signatures.clear();
    }

    /// Get required signatures
    pub fn required_signatures(&self) -> usize {
        self.threshold
    }

    /// Get authorized relayers
    pub fn get_relayers(&self) -> &[Vec<u8>] {
        &self.relayers
    }

    /// Check if pubkey is authorized relayer
    pub fn is_authorized(&self, pubkey: &[u8]) -> bool {
        self.relayers.contains(&pubkey.to_vec())
    }

    /// Get missing signatures count
    pub fn missing_signatures(&self) -> usize {
        self.threshold.saturating_sub(self.current_signatures.len())
    }
}

/// Relayer set management
pub struct RelayerSetManager {
    active_set: MultiSigRelayer,
    pending_set: Option<MultiSigRelayer>,
    update_epoch: u64,
}

impl RelayerSetManager {
    pub fn new(relayers: Vec<Vec<u8>>, threshold: usize) -> Result<Self, MultiSigError> {
        let active_set = MultiSigRelayer::new(relayers, threshold)?;
        
        Ok(Self {
            active_set,
            pending_set: None,
            update_epoch: 0,
        })
    }

    /// Propose new relayer set
    pub fn propose_update(
        &mut self,
        new_relayers: Vec<Vec<u8>>,
        new_threshold: usize,
        activation_epoch: u64,
    ) -> Result<(), MultiSigError> {
        let pending_set = MultiSigRelayer::new(new_relayers, new_threshold)?;
        self.pending_set = Some(pending_set);
        self.update_epoch = activation_epoch;
        Ok(())
    }

    /// Activate pending relayer set
    pub fn activate_pending(&mut self, current_epoch: u64) -> Result<()> {
        if current_epoch < self.update_epoch {
            return Err(anyhow::anyhow!("Update epoch not reached"));
        }

        if let Some(pending) = self.pending_set.take() {
            self.active_set = pending;
            self.update_epoch = 0;
        }

        Ok(())
    }

    /// Get active relayer set
    pub fn get_active_set(&self) -> &MultiSigRelayer {
        &self.active_set
    }

    /// Get pending relayer set
    pub fn get_pending_set(&self) -> Option<&MultiSigRelayer> {
        self.pending_set.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::Keypair;
    use ed25519_dalek::Signer;
    use rand::rngs::OsRng;

    #[test]
    fn test_multisig_creation() {
        let relayers = vec![
            vec![1u8; 32],
            vec![2u8; 32],
            vec![3u8; 32],
        ];

        let multisig = MultiSigRelayer::new(relayers, 2);
        assert!(multisig.is_ok());

        let ms = multisig.unwrap();
        assert_eq!(ms.required_signatures(), 2);
    }

    #[test]
    fn test_invalid_threshold() {
        let relayers = vec![vec![1u8; 32], vec![2u8; 32]];
        
        let result = MultiSigRelayer::new(relayers, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_signature_verification() {
        let mut csprng = OsRng;
        
        // Create 3 relayers
        let relayer1 = Keypair::generate(&mut csprng);
        let relayer2 = Keypair::generate(&mut csprng);
        let relayer3 = Keypair::generate(&mut csprng);
        
        let relayers = vec![
            relayer1.public.to_bytes().to_vec(),
            relayer2.public.to_bytes().to_vec(),
            relayer3.public.to_bytes().to_vec(),
        ];

        let mut multisig = MultiSigRelayer::new(relayers, 2).unwrap();

        let message = b"test message";

        // Add signatures from relayer1 and relayer2
        let sig1 = relayer1.sign(message);
        multisig.add_signature(RelayerSignature {
            relayer_pubkey: relayer1.public.to_bytes().to_vec(),
            signature: sig1.to_bytes().to_vec(),
            timestamp: 12345,
        }).unwrap();

        let sig2 = relayer2.sign(message);
        multisig.add_signature(RelayerSignature {
            relayer_pubkey: relayer2.public.to_bytes().to_vec(),
            signature: sig2.to_bytes().to_vec(),
            timestamp: 12345,
        }).unwrap();

        assert!(multisig.has_threshold());
        assert!(multisig.verify_signatures(message).is_ok());
    }

    #[test]
    fn test_unauthorized_relayer() {
        let relayers = vec![vec![1u8; 32], vec![2u8; 32]];
        let mut multisig = MultiSigRelayer::new(relayers, 2).unwrap();

        let unauthorized_sig = RelayerSignature {
            relayer_pubkey: vec![99u8; 32],
            signature: vec![0u8; 64],
            timestamp: 12345,
        };

        let result = multisig.add_signature(unauthorized_sig);
        assert!(matches!(result, Err(MultiSigError::UnauthorizedRelayer(_))));
    }

    #[test]
    fn test_duplicate_signature() {
        let relayers = vec![vec![1u8; 32]];
        let mut multisig = MultiSigRelayer::new(relayers, 1).unwrap();

        let sig = RelayerSignature {
            relayer_pubkey: vec![1u8; 32],
            signature: vec![0u8; 64],
            timestamp: 12345,
        };

        multisig.add_signature(sig.clone()).unwrap();
        let result = multisig.add_signature(sig);
        
        assert!(matches!(result, Err(MultiSigError::DuplicateSignature(_))));
    }

    #[test]
    fn test_relayer_set_manager() {
        let relayers = vec![vec![1u8; 32], vec![2u8; 32]];
        let manager = RelayerSetManager::new(relayers.clone(), 2).unwrap();
        
        assert_eq!(manager.get_active_set().required_signatures(), 2);
    }

    #[test]
    fn test_relayer_set_update() {
        let relayers = vec![vec![1u8; 32], vec![2u8; 32]];
        let mut manager = RelayerSetManager::new(relayers, 2).unwrap();
        
        let new_relayers = vec![vec![3u8; 32], vec![4u8; 32], vec![5u8; 32]];
        manager.propose_update(new_relayers, 2, 10).unwrap();
        
        assert!(manager.get_pending_set().is_some());
        
        // Activate at epoch 10
        manager.activate_pending(10).unwrap();
        
        assert_eq!(manager.get_active_set().get_relayers().len(), 3);
        assert!(manager.get_pending_set().is_none());
    }
}







