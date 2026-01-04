use thiserror::Error;

/// Common error types used across the node
#[derive(Error, Debug)]
pub enum NodeError {
    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Timeout error: {0}")]
    Timeout(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Already exists: {0}")]
    AlreadyExists(String),

    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("Capacity exceeded: {0}")]
    CapacityExceeded(String),

    #[error("Configuration error: {0}")]
    Configuration(String),
}

impl From<std::io::Error> for NodeError {
    fn from(err: std::io::Error) -> Self {
        NodeError::Storage(err.to_string())
    }
}

impl From<bincode::Error> for NodeError {
    fn from(err: bincode::Error) -> Self {
        NodeError::Storage(err.to_string())
    }
}

/// AI Runtime specific errors
#[derive(Error, Debug)]
pub enum AIRuntimeError {
    #[error("Agent not found: {0}")]
    AgentNotFound(String),

    #[error("Agent already exists: {0}")]
    AgentAlreadyExists(String),

    #[error("Agent not active: status is {0:?}")]
    AgentNotActive(String),

    #[error("Compute budget exceeded: {requested} > {budget}")]
    ComputeBudgetExceeded { requested: u64, budget: u64 },

    #[error("Unauthorized access: {0}")]
    Unauthorized(String),

    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Bridge specific errors
#[derive(Error, Debug)]
pub enum BridgeError {
    #[error("Operation not found: {0}")]
    OperationNotFound(String),

    #[error("Unsupported token: {0}")]
    UnsupportedToken(String),

    #[error("Invalid amount: {0}")]
    InvalidAmount(String),

    #[error("Insufficient signatures: have {have}, need {need}")]
    InsufficientSignatures { have: usize, need: usize },

    #[error("Proof verification failed: {0}")]
    ProofVerificationFailed(String),

    #[error("Already processed: {0}")]
    AlreadyProcessed(String),

    #[error("Chain not supported: {0}")]
    ChainNotSupported(String),
}

/// Mempool specific errors
#[derive(Error, Debug)]
pub enum MempoolError {
    #[error("Transaction already in mempool")]
    DuplicateTransaction,

    #[error("Fee too low: {fee} < {min_fee}")]
    FeeTooLow { fee: u64, min_fee: u64 },

    #[error("Account transaction limit exceeded")]
    AccountLimitExceeded,

    #[error("Mempool full: {current} >= {max}")]
    MempoolFull { current: usize, max: usize },

    #[error("Transaction too large: {size} > {max}")]
    TransactionTooLarge { size: usize, max: usize },
}

/// Convert Result<T, NodeError> to Result<T, anyhow::Error>
pub type NodeResult<T> = Result<T, NodeError>;
pub type AIResult<T> = Result<T, AIRuntimeError>;
pub type BridgeResult<T> = Result<T, BridgeError>;
pub type MempoolResult<T> = Result<T, MempoolError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = NodeError::NotFound("test".to_string());
        assert_eq!(err.to_string(), "Not found: test");

        let err = AIRuntimeError::ComputeBudgetExceeded {
            requested: 100000,
            budget: 50000,
        };
        assert!(err.to_string().contains("100000"));
    }

    #[test]
    fn test_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let node_err: NodeError = io_err.into();
        
        assert!(matches!(node_err, NodeError::Storage(_)));
    }
}

