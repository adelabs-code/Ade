use serde::{Serialize, Deserialize};
use anyhow::Result;

/// Account rent system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RentCollector {
    pub lamports_per_byte_year: u64,
    pub exemption_threshold: f64,
    pub burn_percent: u8,
}

impl Default for RentCollector {
    fn default() -> Self {
        Self {
            lamports_per_byte_year: 3_480,
            exemption_threshold: 2.0,  // 2 years of rent
            burn_percent: 50,
        }
    }
}

impl RentCollector {
    /// Calculate rent for account
    pub fn calculate_rent(&self, data_size: usize, epochs: u64) -> u64 {
        let bytes = data_size as u64;
        let years = epochs as f64 / 365.25;
        
        (self.lamports_per_byte_year * bytes) as f64 * years) as u64
    }

    /// Calculate minimum balance for rent exemption
    pub fn calculate_rent_exempt_balance(&self, data_size: usize) -> u64 {
        let rent_per_year = self.lamports_per_byte_year * data_size as u64;
        (rent_per_year as f64 * self.exemption_threshold) as u64
    }

    /// Check if account is rent exempt
    pub fn is_rent_exempt(&self, balance: u64, data_size: usize) -> bool {
        balance >= self.calculate_rent_exempt_balance(data_size)
    }

    /// Collect rent from account
    pub fn collect_rent(&self, balance: u64, data_size: usize, epochs_elapsed: u64) -> RentResult {
        let exempt_balance = self.calculate_rent_exempt_balance(data_size);
        
        if balance >= exempt_balance {
            return RentResult {
                rent_due: 0,
                rent_collected: 0,
                account_deleted: false,
                new_balance: balance,
            };
        }

        let rent_due = self.calculate_rent(data_size, epochs_elapsed);
        let rent_collected = rent_due.min(balance);
        let new_balance = balance.saturating_sub(rent_collected);
        
        // Delete account if balance reaches zero
        let account_deleted = new_balance == 0 && data_size > 0;

        RentResult {
            rent_due,
            rent_collected,
            account_deleted,
            new_balance,
        }
    }

    /// Calculate rent for next epoch
    pub fn rent_due_for_next_epoch(&self, balance: u64, data_size: usize) -> u64 {
        if self.is_rent_exempt(balance, data_size) {
            0
        } else {
            self.calculate_rent(data_size, 1)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RentResult {
    pub rent_due: u64,
    pub rent_collected: u64,
    pub account_deleted: bool,
    pub new_balance: u64,
}

/// Account lifecycle manager
pub struct AccountLifecycle {
    rent_collector: RentCollector,
    current_epoch: u64,
}

impl AccountLifecycle {
    pub fn new(rent_collector: RentCollector) -> Self {
        Self {
            rent_collector,
            current_epoch: 0,
        }
    }

    /// Process account for current epoch
    pub fn process_account(
        &self,
        balance: u64,
        data_size: usize,
        last_epoch: u64,
    ) -> RentResult {
        let epochs_elapsed = self.current_epoch.saturating_sub(last_epoch);
        
        if epochs_elapsed == 0 {
            return RentResult {
                rent_due: 0,
                rent_collected: 0,
                account_deleted: false,
                new_balance: balance,
            };
        }

        self.rent_collector.collect_rent(balance, data_size, epochs_elapsed)
    }

    /// Advance to next epoch
    pub fn advance_epoch(&mut self) {
        self.current_epoch += 1;
    }

    /// Get current epoch
    pub fn current_epoch(&self) -> u64 {
        self.current_epoch
    }

    /// Check if account should be deleted
    pub fn should_delete_account(&self, balance: u64, data_size: usize) -> bool {
        balance == 0 && data_size > 0
    }

    /// Get minimum balance to create account
    pub fn minimum_balance(&self, data_size: usize) -> u64 {
        self.rent_collector.calculate_rent_exempt_balance(data_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rent_calculation() {
        let collector = RentCollector::default();
        
        let rent = collector.calculate_rent(100, 1);
        assert_eq!(rent, 348_000); // 100 bytes * 3480 lamports/byte/year * 1 year
    }

    #[test]
    fn test_rent_exemption() {
        let collector = RentCollector::default();
        
        let data_size = 100;
        let exempt_balance = collector.calculate_rent_exempt_balance(data_size);
        
        assert!(collector.is_rent_exempt(exempt_balance, data_size));
        assert!(!collector.is_rent_exempt(exempt_balance - 1, data_size));
    }

    #[test]
    fn test_rent_collection() {
        let collector = RentCollector::default();
        
        let result = collector.collect_rent(100_000, 100, 1);
        
        assert!(result.rent_collected > 0);
        assert_eq!(result.new_balance, 100_000 - result.rent_collected);
        assert!(!result.account_deleted);
    }

    #[test]
    fn test_account_deletion() {
        let collector = RentCollector::default();
        
        let result = collector.collect_rent(1000, 100, 1);
        
        // Rent for 100 bytes for 1 epoch should exceed 1000 lamports
        assert!(result.account_deleted);
        assert_eq!(result.new_balance, 0);
    }

    #[test]
    fn test_exempt_account_no_rent() {
        let collector = RentCollector::default();
        let data_size = 100;
        
        let exempt_balance = collector.calculate_rent_exempt_balance(data_size);
        let result = collector.collect_rent(exempt_balance, data_size, 10);
        
        assert_eq!(result.rent_collected, 0);
        assert_eq!(result.new_balance, exempt_balance);
    }

    #[test]
    fn test_account_lifecycle() {
        let mut lifecycle = AccountLifecycle::new(RentCollector::default());
        
        lifecycle.advance_epoch();
        assert_eq!(lifecycle.current_epoch(), 1);
        
        let result = lifecycle.process_account(1_000_000, 100, 0);
        assert!(result.rent_collected > 0);
    }

    #[test]
    fn test_no_rent_same_epoch() {
        let lifecycle = AccountLifecycle::new(RentCollector::default());
        
        let result = lifecycle.process_account(1_000_000, 100, 0);
        assert_eq!(result.rent_collected, 0);
    }

    #[test]
    fn test_minimum_balance() {
        let lifecycle = AccountLifecycle::new(RentCollector::default());
        
        let min_balance = lifecycle.minimum_balance(100);
        assert!(min_balance > 0);
        
        let collector = &lifecycle.rent_collector;
        assert!(collector.is_rent_exempt(min_balance, 100));
    }
}

