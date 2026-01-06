use anyhow::{Result, Context};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Compute unit costs for different operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeCosts {
    pub base_cost: u64,
    pub per_byte_cost: u64,
    pub operation_costs: HashMap<String, u64>,
}

impl Default for ComputeCosts {
    fn default() -> Self {
        let mut operation_costs = HashMap::new();
        
        // Base operations
        operation_costs.insert("sha256".to_string(), 20);
        operation_costs.insert("ed25519_verify".to_string(), 3000);
        operation_costs.insert("secp256k1_recover".to_string(), 25000);
        
        // AI operations
        operation_costs.insert("model_load".to_string(), 10000);
        operation_costs.insert("inference_base".to_string(), 1000);
        operation_costs.insert("inference_per_token".to_string(), 100);
        operation_costs.insert("embeddings_base".to_string(), 500);
        operation_costs.insert("embeddings_per_token".to_string(), 50);
        
        // Storage operations
        operation_costs.insert("account_read".to_string(), 100);
        operation_costs.insert("account_write".to_string(), 200);
        
        // Bridge operations
        operation_costs.insert("bridge_lock".to_string(), 5000);
        operation_costs.insert("bridge_unlock".to_string(), 5000);
        operation_costs.insert("merkle_verify".to_string(), 500);
        
        Self {
            base_cost: 1000,
            per_byte_cost: 1,
            operation_costs,
        }
    }
}

/// Compute meter for tracking resource usage
pub struct ComputeMeter {
    costs: ComputeCosts,
    consumed: u64,
    budget: u64,
    operations: Vec<ComputeOperation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeOperation {
    pub operation: String,
    pub cost: u64,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub enum ComputeError {
    BudgetExceeded { consumed: u64, budget: u64 },
    InvalidOperation(String),
}

impl std::fmt::Display for ComputeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::BudgetExceeded { consumed, budget } => {
                write!(f, "Compute budget exceeded: {} > {}", consumed, budget)
            }
            Self::InvalidOperation(op) => {
                write!(f, "Invalid operation: {}", op)
            }
        }
    }
}

impl std::error::Error for ComputeError {}

impl ComputeMeter {
    pub fn new(budget: u64) -> Self {
        Self::with_costs(budget, ComputeCosts::default())
    }

    pub fn with_costs(budget: u64, costs: ComputeCosts) -> Self {
        Self {
            costs,
            consumed: 0,
            budget,
            operations: Vec::new(),
        }
    }

    /// Consume compute units for an operation
    pub fn consume(&mut self, operation: &str) -> Result<(), ComputeError> {
        let cost = self.costs.operation_costs.get(operation)
            .copied()
            .ok_or_else(|| ComputeError::InvalidOperation(operation.to_string()))?;

        self.consume_units(cost, operation)
    }

    /// Consume specific number of compute units
    pub fn consume_units(&mut self, units: u64, operation: &str) -> Result<(), ComputeError> {
        let new_consumed = self.consumed + units;
        
        if new_consumed > self.budget {
            return Err(ComputeError::BudgetExceeded {
                consumed: new_consumed,
                budget: self.budget,
            });
        }

        self.consumed = new_consumed;
        
        self.operations.push(ComputeOperation {
            operation: operation.to_string(),
            cost: units,
            timestamp: current_timestamp(),
        });

        Ok(())
    }

    /// Consume compute based on data size
    pub fn consume_for_data(&mut self, data_len: usize, operation: &str) -> Result<(), ComputeError> {
        let cost = self.costs.per_byte_cost * data_len as u64;
        self.consume_units(cost, operation)
    }

    /// Get remaining compute budget
    pub fn remaining(&self) -> u64 {
        self.budget.saturating_sub(self.consumed)
    }

    /// Get consumed compute
    pub fn consumed(&self) -> u64 {
        self.consumed
    }

    /// Get budget
    pub fn budget(&self) -> u64 {
        self.budget
    }

    /// Check if operation is affordable
    pub fn can_afford(&self, operation: &str) -> bool {
        if let Some(&cost) = self.costs.operation_costs.get(operation) {
            self.consumed + cost <= self.budget
        } else {
            false
        }
    }

    /// Get operation history
    pub fn get_operations(&self) -> &[ComputeOperation] {
        &self.operations
    }

    /// Reset meter
    pub fn reset(&mut self) {
        self.consumed = 0;
        self.operations.clear();
    }

    /// Get meter statistics
    pub fn get_stats(&self) -> ComputeStats {
        let operation_counts = self.operations.iter()
            .fold(HashMap::new(), |mut acc, op| {
                *acc.entry(op.operation.clone()).or_insert(0) += 1;
                acc
            });

        ComputeStats {
            budget: self.budget,
            consumed: self.consumed,
            remaining: self.remaining(),
            operation_count: self.operations.len(),
            operation_counts,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeStats {
    pub budget: u64,
    pub consumed: u64,
    pub remaining: u64,
    pub operation_count: usize,
    pub operation_counts: HashMap<String, usize>,
}

/// AI-specific compute estimation
pub struct AIComputeEstimator;

impl AIComputeEstimator {
    /// Estimate compute for inference
    pub fn estimate_inference(
        model_type: &str,
        input_tokens: usize,
        max_output_tokens: usize,
    ) -> u64 {
        let base_costs = match model_type {
            "transformer" => (10000, 100),
            "cnn" => (5000, 50),
            "rnn" => (7000, 75),
            _ => (8000, 80),
        };

        let (model_load, per_token) = base_costs;
        
        model_load +
        (input_tokens as u64 * per_token) +
        (max_output_tokens as u64 * per_token)
    }

    /// Estimate compute for embeddings
    pub fn estimate_embeddings(
        input_tokens: usize,
        embedding_dim: usize,
    ) -> u64 {
        let base = 500u64;
        let per_token = 50u64;
        let dim_cost = (embedding_dim as u64 / 100).max(1);
        
        base + (input_tokens as u64 * per_token * dim_cost)
    }

    /// Estimate compute for fine-tuning
    pub fn estimate_fine_tuning(
        dataset_size: usize,
        epochs: usize,
        model_size_mb: usize,
    ) -> u64 {
        let base = 50000u64;
        let per_sample = 1000u64;
        let model_factor = model_size_mb as u64 * 100;
        
        base + (dataset_size as u64 * per_sample * epochs as u64) + model_factor
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
    fn test_compute_meter_basic() {
        let mut meter = ComputeMeter::new(100000);
        
        assert!(meter.consume("sha256").is_ok());
        assert_eq!(meter.consumed(), 20);
        assert_eq!(meter.remaining(), 99980);
    }

    #[test]
    fn test_budget_exceeded() {
        let mut meter = ComputeMeter::new(1000);
        
        // Consume more than budget
        meter.consume("model_load").unwrap();
        let result = meter.consume("inference_base");
        
        assert!(matches!(result, Err(ComputeError::BudgetExceeded { .. })));
    }

    #[test]
    fn test_consume_for_data() {
        let mut meter = ComputeMeter::new(100000);
        
        meter.consume_for_data(1000, "data_processing").unwrap();
        assert_eq!(meter.consumed(), 1000);
    }

    #[test]
    fn test_can_afford() {
        let mut meter = ComputeMeter::new(5000);
        
        assert!(meter.can_afford("sha256"));
        assert!(!meter.can_afford("model_load"));
    }

    #[test]
    fn test_operation_tracking() {
        let mut meter = ComputeMeter::new(100000);
        
        meter.consume("sha256").unwrap();
        meter.consume("ed25519_verify").unwrap();
        
        let ops = meter.get_operations();
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].operation, "sha256");
    }

    #[test]
    fn test_meter_reset() {
        let mut meter = ComputeMeter::new(100000);
        
        meter.consume("sha256").unwrap();
        assert!(meter.consumed() > 0);
        
        meter.reset();
        assert_eq!(meter.consumed(), 0);
        assert_eq!(meter.get_operations().len(), 0);
    }

    #[test]
    fn test_ai_inference_estimation() {
        let cost = AIComputeEstimator::estimate_inference("transformer", 50, 200);
        
        // model_load (10000) + input (50*100) + output (200*100)
        assert_eq!(cost, 10000 + 5000 + 20000);
    }

    #[test]
    fn test_embeddings_estimation() {
        let cost = AIComputeEstimator::estimate_embeddings(100, 768);
        
        // base (500) + tokens (100*50*7.68)
        assert!(cost > 500);
    }

    #[test]
    fn test_compute_stats() {
        let mut meter = ComputeMeter::new(100000);
        
        meter.consume("sha256").unwrap();
        meter.consume("sha256").unwrap();
        meter.consume("ed25519_verify").unwrap();
        
        let stats = meter.get_stats();
        assert_eq!(stats.operation_count, 3);
        assert_eq!(stats.operation_counts.get("sha256"), Some(&2));
        assert_eq!(stats.operation_counts.get("ed25519_verify"), Some(&1));
    }
}







