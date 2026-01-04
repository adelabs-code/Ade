pub mod bridge;
pub mod relayer;
pub mod lock_contract;
pub mod merkle;
pub mod multisig;

// Re-export Solana program types
pub mod solana_lock {
    pub use super::lock_contract::*;
    
    // Include BridgeProof for cross-module use
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
