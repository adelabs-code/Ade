use thiserror::Error;
use crate::transaction::Transaction;
use crate::instruction::InstructionType;

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Invalid signature")]
    InvalidSignature,
    
    #[error("Insufficient balance: required {required}, available {available}")]
    InsufficientBalance { required: u64, available: u64 },
    
    #[error("Invalid account: {0}")]
    InvalidAccount(String),
    
    #[error("Invalid instruction data")]
    InvalidInstructionData,
    
    #[error("Compute budget exceeded: {0}")]
    ComputeBudgetExceeded(u64),
    
    #[error("Invalid AI model hash")]
    InvalidModelHash,
    
    #[error("Unauthorized access")]
    Unauthorized,
}

pub struct TransactionValidator {
    max_compute_units: u64,
    max_transaction_size: usize,
}

impl TransactionValidator {
    pub fn new() -> Self {
        Self {
            max_compute_units: 1_400_000,
            max_transaction_size: 1232,
        }
    }

    pub fn with_limits(max_compute_units: u64, max_transaction_size: usize) -> Self {
        Self {
            max_compute_units,
            max_transaction_size,
        }
    }

    pub fn validate(&self, transaction: &Transaction) -> Result<(), ValidationError> {
        self.validate_signatures(transaction)?;
        self.validate_size(transaction)?;
        self.validate_instructions(transaction)?;
        Ok(())
    }

    fn validate_signatures(&self, transaction: &Transaction) -> Result<(), ValidationError> {
        transaction.verify()
            .map_err(|_| ValidationError::InvalidSignature)?;
        Ok(())
    }

    fn validate_size(&self, transaction: &Transaction) -> Result<(), ValidationError> {
        let serialized = transaction.serialize()
            .map_err(|_| ValidationError::InvalidInstructionData)?;
        
        if serialized.len() > self.max_transaction_size {
            return Err(ValidationError::InvalidAccount(
                format!("Transaction size {} exceeds maximum {}", 
                    serialized.len(), self.max_transaction_size)
            ));
        }

        Ok(())
    }

    fn validate_instructions(&self, transaction: &Transaction) -> Result<(), ValidationError> {
        let mut total_compute = 0u64;

        for instruction in &transaction.message.instructions {
            if let Ok(ix_type) = bincode::deserialize::<InstructionType>(&instruction.data) {
                match ix_type {
                    InstructionType::AIAgentExecute { max_compute, .. } => {
                        total_compute += max_compute;
                        if total_compute > self.max_compute_units {
                            return Err(ValidationError::ComputeBudgetExceeded(total_compute));
                        }
                    }
                    InstructionType::Transfer { amount, .. } => {
                        if amount == 0 {
                            return Err(ValidationError::InvalidInstructionData);
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(())
    }
}

impl Default for TransactionValidator {
    fn default() -> Self {
        Self::new()
    }
}

