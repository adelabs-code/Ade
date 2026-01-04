use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use serde::{Serialize, Deserialize};
use anyhow::Result;

/// Dynamic fee market based on network congestion
pub struct FeeMarket {
    recent_blocks: Arc<RwLock<VecDeque<BlockFeeData>>>,
    base_fee: Arc<RwLock<u64>>,
    config: FeeMarketConfig,
}

#[derive(Debug, Clone)]
pub struct FeeMarketConfig {
    pub target_block_utilization: f64,  // 0.5 = 50%
    pub max_fee_increase: f64,           // Max fee increase per block
    pub max_fee_decrease: f64,           // Max fee decrease per block
    pub min_base_fee: u64,
    pub max_base_fee: u64,
    pub history_size: usize,
}

impl Default for FeeMarketConfig {
    fn default() -> Self {
        Self {
            target_block_utilization: 0.5,
            max_fee_increase: 0.125,  // 12.5%
            max_fee_decrease: 0.125,
            min_base_fee: 5000,
            max_base_fee: 1_000_000,
            history_size: 150,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockFeeData {
    pub slot: u64,
    pub base_fee: u64,
    pub transaction_count: usize,
    pub max_transactions: usize,
    pub utilization: f64,
    pub min_priority_fee: u64,
    pub max_priority_fee: u64,
    pub median_priority_fee: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeEstimate {
    pub min_fee: u64,
    pub low_fee: u64,
    pub medium_fee: u64,
    pub high_fee: u64,
    pub base_fee: u64,
    pub congestion_multiplier: f64,
}

impl FeeMarket {
    pub fn new(config: FeeMarketConfig) -> Self {
        let base_fee = config.min_base_fee;
        
        Self {
            recent_blocks: Arc::new(RwLock::new(VecDeque::new())),
            base_fee: Arc::new(RwLock::new(base_fee)),
            config,
        }
    }

    /// Record block data and adjust base fee
    pub fn record_block(
        &self,
        slot: u64,
        transaction_count: usize,
        max_transactions: usize,
        priority_fees: Vec<u64>,
    ) {
        let utilization = transaction_count as f64 / max_transactions as f64;
        let current_base_fee = *self.base_fee.read().unwrap();

        let (min_priority, max_priority, median_priority) = if !priority_fees.is_empty() {
            let mut sorted = priority_fees.clone();
            sorted.sort_unstable();
            
            let min = *sorted.first().unwrap();
            let max = *sorted.last().unwrap();
            let median = sorted[sorted.len() / 2];
            
            (min, max, median)
        } else {
            (0, 0, 0)
        };

        let block_data = BlockFeeData {
            slot,
            base_fee: current_base_fee,
            transaction_count,
            max_transactions,
            utilization,
            min_priority_fee: min_priority,
            max_priority_fee: max_priority,
            median_priority_fee: median_priority,
        };

        // Add to history
        {
            let mut history = self.recent_blocks.write().unwrap();
            history.push_back(block_data);
            
            if history.len() > self.config.history_size {
                history.pop_front();
            }
        }

        // Adjust base fee based on utilization
        self.adjust_base_fee(utilization);
    }

    /// Adjust base fee based on network utilization
    fn adjust_base_fee(&self, utilization: f64) {
        let mut base_fee = self.base_fee.write().unwrap();
        let current_fee = *base_fee;

        let adjustment = if utilization > self.config.target_block_utilization {
            // Increase fee
            let increase = current_fee as f64 * self.config.max_fee_increase;
            let excess_util = utilization - self.config.target_block_utilization;
            let scale = (excess_util / (1.0 - self.config.target_block_utilization)).min(1.0);
            (increase * scale) as u64
        } else {
            // Decrease fee
            let decrease = current_fee as f64 * self.config.max_fee_decrease;
            let under_util = self.config.target_block_utilization - utilization;
            let scale = (under_util / self.config.target_block_utilization).min(1.0);
            -(decrease * scale) as i64
        };

        let new_fee = if adjustment >= 0 {
            current_fee + adjustment as u64
        } else {
            current_fee.saturating_sub(adjustment.unsigned_abs())
        };

        // Clamp to min/max
        *base_fee = new_fee.clamp(self.config.min_base_fee, self.config.max_base_fee);
    }

    /// Get current base fee
    pub fn get_base_fee(&self) -> u64 {
        *self.base_fee.read().unwrap()
    }

    /// Estimate fees for different priority levels
    pub fn estimate_fees(&self) -> FeeEstimate {
        let base_fee = self.get_base_fee();
        let history = self.recent_blocks.read().unwrap();

        // Calculate congestion based on recent blocks
        let avg_utilization = if !history.is_empty() {
            history.iter().map(|b| b.utilization).sum::<f64>() / history.len() as f64
        } else {
            0.5
        };

        let congestion_multiplier = if avg_utilization > 0.8 {
            2.0
        } else if avg_utilization > 0.6 {
            1.5
        } else if avg_utilization > 0.4 {
            1.0
        } else {
            0.8
        };

        // Get recent priority fees
        let recent_priority_fees: Vec<u64> = history.iter()
            .map(|b| b.median_priority_fee)
            .collect();

        let median_priority = if !recent_priority_fees.is_empty() {
            let mut sorted = recent_priority_fees.clone();
            sorted.sort_unstable();
            sorted[sorted.len() / 2]
        } else {
            0
        };

        FeeEstimate {
            min_fee: base_fee,
            low_fee: base_fee + (median_priority as f64 * 0.5) as u64,
            medium_fee: base_fee + median_priority,
            high_fee: base_fee + (median_priority as f64 * 2.0) as u64,
            base_fee,
            congestion_multiplier,
        }
    }

    /// Get recommended priority fee for target inclusion time
    pub fn get_priority_fee_for_target(&self, target_blocks: u64) -> u64 {
        let estimate = self.estimate_fees();
        
        match target_blocks {
            1 => estimate.high_fee - estimate.base_fee,
            2..=3 => estimate.medium_fee - estimate.base_fee,
            4..=10 => estimate.low_fee - estimate.base_fee,
            _ => 0,
        }
    }

    /// Get recent fee history
    pub fn get_recent_history(&self, count: usize) -> Vec<BlockFeeData> {
        let history = self.recent_blocks.read().unwrap();
        history.iter()
            .rev()
            .take(count)
            .cloned()
            .collect()
    }

    /// Get average base fee over recent blocks
    pub fn get_average_base_fee(&self, blocks: usize) -> u64 {
        let history = self.recent_blocks.read().unwrap();
        
        if history.is_empty() {
            return self.get_base_fee();
        }

        let recent: Vec<_> = history.iter().rev().take(blocks).collect();
        let sum: u64 = recent.iter().map(|b| b.base_fee).sum();
        
        sum / recent.len() as u64
    }

    /// Get fee market statistics
    pub fn get_stats(&self) -> FeeMarketStats {
        let history = self.recent_blocks.read().unwrap();
        let base_fee = self.get_base_fee();

        let avg_utilization = if !history.is_empty() {
            history.iter().map(|b| b.utilization).sum::<f64>() / history.len() as f64
        } else {
            0.0
        };

        let avg_tx_count = if !history.is_empty() {
            history.iter().map(|b| b.transaction_count).sum::<usize>() / history.len()
        } else {
            0
        };

        FeeMarketStats {
            current_base_fee: base_fee,
            average_utilization: avg_utilization,
            average_transactions_per_block: avg_tx_count,
            blocks_tracked: history.len(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeMarketStats {
    pub current_base_fee: u64,
    pub average_utilization: f64,
    pub average_transactions_per_block: usize,
    pub blocks_tracked: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fee_market_creation() {
        let market = FeeMarket::new(FeeMarketConfig::default());
        assert_eq!(market.get_base_fee(), 5000);
    }

    #[test]
    fn test_fee_increase_on_congestion() {
        let market = FeeMarket::new(FeeMarketConfig::default());
        let initial_fee = market.get_base_fee();
        
        // Record high utilization blocks
        for i in 0..10 {
            market.record_block(i, 9000, 10000, vec![]);
        }
        
        let new_fee = market.get_base_fee();
        assert!(new_fee > initial_fee);
    }

    #[test]
    fn test_fee_decrease_on_low_usage() {
        let market = FeeMarket::new(FeeMarketConfig::default());
        
        // Set higher initial fee
        {
            let mut base_fee = market.base_fee.write().unwrap();
            *base_fee = 50000;
        }
        
        let initial_fee = market.get_base_fee();
        
        // Record low utilization blocks
        for i in 0..10 {
            market.record_block(i, 1000, 10000, vec![]);
        }
        
        let new_fee = market.get_base_fee();
        assert!(new_fee < initial_fee);
    }

    #[test]
    fn test_fee_estimation() {
        let market = FeeMarket::new(FeeMarketConfig::default());
        
        // Record some blocks with priority fees
        for i in 0..5 {
            let priority_fees = vec![1000, 2000, 3000, 5000, 10000];
            market.record_block(i, 5000, 10000, priority_fees);
        }
        
        let estimate = market.estimate_fees();
        assert!(estimate.min_fee > 0);
        assert!(estimate.medium_fee > estimate.low_fee);
        assert!(estimate.high_fee > estimate.medium_fee);
    }

    #[test]
    fn test_priority_fee_for_target() {
        let market = FeeMarket::new(FeeMarketConfig::default());
        
        market.record_block(0, 5000, 10000, vec![1000, 2000, 3000]);
        
        let immediate = market.get_priority_fee_for_target(1);
        let fast = market.get_priority_fee_for_target(3);
        let slow = market.get_priority_fee_for_target(10);
        
        assert!(immediate >= fast);
        assert!(fast >= slow);
    }

    #[test]
    fn test_fee_clamping() {
        let market = FeeMarket::new(FeeMarketConfig::default());
        
        // Try to push fee very high
        for i in 0..100 {
            market.record_block(i, 10000, 10000, vec![]);
        }
        
        let fee = market.get_base_fee();
        assert!(fee <= market.config.max_base_fee);
        assert!(fee >= market.config.min_base_fee);
    }

    #[test]
    fn test_history_size_limit() {
        let mut config = FeeMarketConfig::default();
        config.history_size = 5;
        let market = FeeMarket::new(config);
        
        // Add more blocks than history size
        for i in 0..10 {
            market.record_block(i, 5000, 10000, vec![]);
        }
        
        let history = market.get_recent_history(100);
        assert_eq!(history.len(), 5);
    }

    #[test]
    fn test_average_base_fee() {
        let market = FeeMarket::new(FeeMarketConfig::default());
        
        // Set different base fees
        for i in 0..5 {
            *market.base_fee.write().unwrap() = 10000 * (i + 1);
            market.record_block(i, 5000, 10000, vec![]);
        }
        
        let avg = market.get_average_base_fee(5);
        // Average of 10000, 20000, 30000, 40000, 50000 = 30000
        assert_eq!(avg, 30000);
    }
}




