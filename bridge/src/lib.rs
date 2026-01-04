pub mod bridge;
pub mod relayer;
pub mod lock_contract;
pub mod merkle;
pub mod multisig;
pub mod proof_verification;
pub mod event_parser;
pub mod bridge_coordinator;

// Re-export Solana program types
pub mod solana_lock {
    pub use super::lock_contract::*;
    
    use serde::{Serialize, Deserialize};
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BridgeProof {
        pub source_tx_hash: Vec<u8>,
        pub block_number: u64,
        pub merkle_proof: Vec<Vec<u8>>,
        pub event_data: Vec<u8>,
        pub relayer_signatures: Vec<Vec<u8>>,
    }
}

pub use bridge::{Bridge, BridgeConfig, BridgeEvent, BridgeStats, DepositInfo, WithdrawalInfo, BridgeStatus};
pub use relayer::{Relayer, RelayerConfig, RelayerStats};
pub use lock_contract::{LockContract, DepositEvent, WithdrawalEvent};
pub use merkle::{MerkleTree, MerkleProof, SparseMerkleTree};
pub use multisig::{MultiSigRelayer, RelayerSignature, MultiSigError, RelayerSetManager};
pub use proof_verification::{ProofVerifier, ProofBuilder, VerificationResult};
pub use event_parser::{EventParser, EventEmitter, ParsedDepositEvent, ParsedWithdrawalEvent, EmittedEvent, EventType};
pub use bridge_coordinator::{BridgeCoordinator, CoordinatorStats};
