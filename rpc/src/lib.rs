pub mod server;
pub mod handlers;
pub mod types;
pub mod state;
pub mod websocket;

/// Re-export AI runtime types for use by handlers
pub mod ai_runtime_types {
    pub use crate::state::{AIRuntimeInterface, ExecutionRequest, ExecutionResult};
}

pub use server::{RpcServer, RpcState};
pub use types::{RpcRequest, RpcResponse, RpcError};
pub use state::{
    RpcStateBackend, ChainState, TransactionInfo, TransactionStatus, AIAgentData,
    BridgeOperationState, BridgeStatus, ExecutionRecord,
    AIRuntimeInterface, ExecutionRequest, ExecutionResult,
};
pub use websocket::{SubscriptionManager, SlotUpdate, AccountUpdate, TransactionUpdate, BlockUpdate};
