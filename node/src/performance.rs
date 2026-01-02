use std::sync::{Arc, RwLock};
use std::collections::VecDeque;
use std::time::Instant;
use serde::{Serialize, Deserialize};

use crate::metrics::MetricsCollector;

/// Performance tracker
pub struct PerformanceTracker {
    metrics: Arc<MetricsCollector>,
    samples: Arc<RwLock<VecDeque<PerformanceSample>>>,
    max_samples: usize,
    start_time: Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSample {
    pub slot: u64,
    pub timestamp: u64,
    pub tps: f64,
    pub block_time_ms: u64,
    pub transaction_count: usize,
    pub compute_units: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub uptime_seconds: u64,
    pub total_transactions_processed: u64,
    pub average_tps: f64,
    pub peak_tps: f64,
    pub average_block_time_ms: u64,
    pub average_compute_per_tx: u64,
    pub total_blocks_produced: u64,
}

impl PerformanceTracker {
    pub fn new(metrics: Arc<MetricsCollector>, max_samples: usize) -> Self {
        Self {
            metrics,
            samples: Arc::new(RwLock::new(VecDeque::new())),
            max_samples,
            start_time: Instant::now(),
        }
    }

    /// Record performance sample
    pub fn record_sample(&self, sample: PerformanceSample) {
        // Update metrics
        self.metrics.set_gauge("tps", sample.tps);
        self.metrics.set_gauge("block_time_ms", sample.block_time_ms as f64);
        self.metrics.increment("total_transactions", sample.transaction_count as u64);
        self.metrics.record_histogram("transaction_count_per_block", sample.transaction_count as f64);
        self.metrics.record_histogram("compute_units_per_block", sample.compute_units as f64);

        // Store sample
        let mut samples = self.samples.write().unwrap();
        samples.push_back(sample);

        if samples.len() > self.max_samples {
            samples.pop_front();
        }
    }

    /// Calculate TPS from recent samples
    pub fn calculate_tps(&self, window_seconds: u64) -> f64 {
        let samples = self.samples.read().unwrap();
        
        if samples.is_empty() {
            return 0.0;
        }

        let cutoff = current_timestamp() - window_seconds;
        let recent: Vec<_> = samples.iter()
            .filter(|s| s.timestamp >= cutoff)
            .collect();

        if recent.is_empty() {
            return 0.0;
        }

        let total_txs: usize = recent.iter().map(|s| s.transaction_count).sum();
        let time_span = recent.last().unwrap().timestamp - recent.first().unwrap().timestamp;

        if time_span == 0 {
            return total_txs as f64;
        }

        total_txs as f64 / time_span as f64
    }

    /// Get average block time
    pub fn average_block_time(&self, sample_count: usize) -> u64 {
        let samples = self.samples.read().unwrap();
        
        if samples.is_empty() {
            return 0;
        }

        let recent: Vec<_> = samples.iter().rev().take(sample_count).collect();
        let total: u64 = recent.iter().map(|s| s.block_time_ms).sum();

        total / recent.len() as u64
    }

    /// Get performance report
    pub fn get_report(&self) -> PerformanceReport {
        let samples = self.samples.read().unwrap();
        let uptime = self.start_time.elapsed().as_secs();

        let total_transactions: u64 = samples.iter()
            .map(|s| s.transaction_count as u64)
            .sum();

        let average_tps = if uptime > 0 {
            total_transactions as f64 / uptime as f64
        } else {
            0.0
        };

        let peak_tps = samples.iter()
            .map(|s| s.tps)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        let average_block_time = if !samples.is_empty() {
            samples.iter().map(|s| s.block_time_ms).sum::<u64>() / samples.len() as u64
        } else {
            0
        };

        let total_compute: u64 = samples.iter().map(|s| s.compute_units).sum();
        let average_compute_per_tx = if total_transactions > 0 {
            total_compute / total_transactions
        } else {
            0
        };

        PerformanceReport {
            uptime_seconds: uptime,
            total_transactions_processed: total_transactions,
            average_tps,
            peak_tps,
            average_block_time_ms: average_block_time,
            average_compute_per_tx,
            total_blocks_produced: samples.len() as u64,
        }
    }

    /// Get recent samples
    pub fn get_recent_samples(&self, count: usize) -> Vec<PerformanceSample> {
        let samples = self.samples.read().unwrap();
        samples.iter().rev().take(count).cloned().collect()
    }

    /// Check if performance is degraded
    pub fn is_degraded(&self) -> bool {
        let avg_block_time = self.average_block_time(10);
        
        // Consider degraded if block time > 1 second
        avg_block_time > 1000
    }

    /// Get uptime
    pub fn uptime_seconds(&self) -> u64 {
        self.start_time.elapsed().as_secs()
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

    fn create_sample(slot: u64, tps: f64, tx_count: usize) -> PerformanceSample {
        PerformanceSample {
            slot,
            timestamp: current_timestamp(),
            tps,
            block_time_ms: 400,
            transaction_count: tx_count,
            compute_units: tx_count as u64 * 1000,
        }
    }

    #[test]
    fn test_performance_tracker() {
        let metrics = Arc::new(MetricsCollector::new());
        let tracker = PerformanceTracker::new(metrics, 100);
        
        tracker.record_sample(create_sample(1, 5000.0, 2000));
        tracker.record_sample(create_sample(2, 6000.0, 2400));
        
        let report = tracker.get_report();
        assert_eq!(report.total_transactions_processed, 4400);
        assert_eq!(report.total_blocks_produced, 2);
    }

    #[test]
    fn test_average_block_time() {
        let metrics = Arc::new(MetricsCollector::new());
        let tracker = PerformanceTracker::new(metrics, 100);
        
        tracker.record_sample(create_sample(1, 5000.0, 2000));
        tracker.record_sample(create_sample(2, 6000.0, 2400));
        tracker.record_sample(create_sample(3, 5500.0, 2200));
        
        let avg = tracker.average_block_time(3);
        assert_eq!(avg, 400);
    }

    #[test]
    fn test_max_samples_limit() {
        let metrics = Arc::new(MetricsCollector::new());
        let tracker = PerformanceTracker::new(metrics, 5);
        
        // Add 10 samples
        for i in 1..=10 {
            tracker.record_sample(create_sample(i, 5000.0, 2000));
        }
        
        let samples = tracker.get_recent_samples(100);
        assert_eq!(samples.len(), 5); // Should only keep last 5
    }

    #[test]
    fn test_degraded_performance() {
        let metrics = Arc::new(MetricsCollector::new());
        let tracker = PerformanceTracker::new(metrics, 100);
        
        // Normal performance
        for i in 0..5 {
            let mut sample = create_sample(i, 5000.0, 2000);
            sample.block_time_ms = 400;
            tracker.record_sample(sample);
        }
        
        assert!(!tracker.is_degraded());
        
        // Degraded performance
        for i in 5..15 {
            let mut sample = create_sample(i, 1000.0, 400);
            sample.block_time_ms = 2000; // 2 seconds
            tracker.record_sample(sample);
        }
        
        assert!(tracker.is_degraded());
    }
}

