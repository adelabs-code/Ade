use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use std::time::{Duration, Instant};

/// Metrics collection system
pub struct MetricsCollector {
    counters: Arc<RwLock<HashMap<String, u64>>>,
    gauges: Arc<RwLock<HashMap<String, f64>>>,
    histograms: Arc<RwLock<HashMap<String, Histogram>>>,
    timers: Arc<RwLock<HashMap<String, Timer>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Histogram {
    values: Vec<f64>,
    count: usize,
    sum: f64,
    min: f64,
    max: f64,
}

impl Histogram {
    fn new() -> Self {
        Self {
            values: Vec::new(),
            count: 0,
            sum: 0.0,
            min: f64::MAX,
            max: f64::MIN,
        }
    }

    fn record(&mut self, value: f64) {
        self.values.push(value);
        self.count += 1;
        self.sum += value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);

        // Keep bounded size
        if self.values.len() > 1000 {
            self.values.remove(0);
        }
    }

    fn percentile(&self, p: f64) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }

        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = ((sorted.len() as f64 - 1.0) * p / 100.0) as usize;
        sorted[index]
    }

    fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }
}

#[derive(Debug, Clone)]
struct Timer {
    start: Option<Instant>,
    duration_ms: Vec<u64>,
}

impl Timer {
    fn new() -> Self {
        Self {
            start: None,
            duration_ms: Vec::new(),
        }
    }

    fn start(&mut self) {
        self.start = Some(Instant::now());
    }

    fn stop(&mut self) -> Option<u64> {
        if let Some(start) = self.start.take() {
            let duration = start.elapsed().as_millis() as u64;
            self.duration_ms.push(duration);
            
            // Keep bounded
            if self.duration_ms.len() > 1000 {
                self.duration_ms.remove(0);
            }
            
            Some(duration)
        } else {
            None
        }
    }

    fn average(&self) -> u64 {
        if self.duration_ms.is_empty() {
            0
        } else {
            self.duration_ms.iter().sum::<u64>() / self.duration_ms.len() as u64
        }
    }
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            counters: Arc::new(RwLock::new(HashMap::new())),
            gauges: Arc::new(RwLock::new(HashMap::new())),
            histograms: Arc::new(RwLock::new(HashMap::new())),
            timers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Increment a counter
    pub fn increment(&self, name: &str, value: u64) {
        let mut counters = self.counters.write().unwrap();
        *counters.entry(name.to_string()).or_insert(0) += value;
    }

    /// Set a gauge value
    pub fn set_gauge(&self, name: &str, value: f64) {
        let mut gauges = self.gauges.write().unwrap();
        gauges.insert(name.to_string(), value);
    }

    /// Record histogram value
    pub fn record_histogram(&self, name: &str, value: f64) {
        let mut histograms = self.histograms.write().unwrap();
        histograms.entry(name.to_string())
            .or_insert_with(Histogram::new)
            .record(value);
    }

    /// Start timer
    pub fn start_timer(&self, name: &str) {
        let mut timers = self.timers.write().unwrap();
        timers.entry(name.to_string())
            .or_insert_with(Timer::new)
            .start();
    }

    /// Stop timer and record duration
    pub fn stop_timer(&self, name: &str) -> Option<u64> {
        let mut timers = self.timers.write().unwrap();
        timers.get_mut(name).and_then(|t| t.stop())
    }

    /// Get counter value
    pub fn get_counter(&self, name: &str) -> u64 {
        let counters = self.counters.read().unwrap();
        counters.get(name).copied().unwrap_or(0)
    }

    /// Get gauge value
    pub fn get_gauge(&self, name: &str) -> f64 {
        let gauges = self.gauges.read().unwrap();
        gauges.get(name).copied().unwrap_or(0.0)
    }

    /// Get histogram stats
    pub fn get_histogram_stats(&self, name: &str) -> Option<HistogramStats> {
        let histograms = self.histograms.read().unwrap();
        histograms.get(name).map(|h| HistogramStats {
            count: h.count,
            mean: h.mean(),
            min: h.min,
            max: h.max,
            p50: h.percentile(50.0),
            p95: h.percentile(95.0),
            p99: h.percentile(99.0),
        })
    }

    /// Get all metrics
    pub fn export_metrics(&self) -> MetricsSnapshot {
        let counters = self.counters.read().unwrap().clone();
        let gauges = self.gauges.read().unwrap().clone();
        
        let histogram_stats: HashMap<String, HistogramStats> = self.histograms.read().unwrap()
            .iter()
            .map(|(name, hist)| {
                (name.clone(), HistogramStats {
                    count: hist.count,
                    mean: hist.mean(),
                    min: hist.min,
                    max: hist.max,
                    p50: hist.percentile(50.0),
                    p95: hist.percentile(95.0),
                    p99: hist.percentile(99.0),
                })
            })
            .collect();

        let timer_stats: HashMap<String, u64> = self.timers.read().unwrap()
            .iter()
            .map(|(name, timer)| (name.clone(), timer.average()))
            .collect();

        MetricsSnapshot {
            counters,
            gauges,
            histograms: histogram_stats,
            timers: timer_stats,
            timestamp: current_timestamp(),
        }
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.counters.write().unwrap().clear();
        self.gauges.write().unwrap().clear();
        self.histograms.write().unwrap().clear();
        self.timers.write().unwrap().clear();
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramStats {
    pub count: usize,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub counters: HashMap<String, u64>,
    pub gauges: HashMap<String, f64>,
    pub histograms: HashMap<String, HistogramStats>,
    pub timers: HashMap<String, u64>,
    pub timestamp: u64,
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
    fn test_counter() {
        let metrics = MetricsCollector::new();
        
        metrics.increment("test_counter", 1);
        metrics.increment("test_counter", 5);
        
        assert_eq!(metrics.get_counter("test_counter"), 6);
    }

    #[test]
    fn test_gauge() {
        let metrics = MetricsCollector::new();
        
        metrics.set_gauge("test_gauge", 42.5);
        assert_eq!(metrics.get_gauge("test_gauge"), 42.5);
        
        metrics.set_gauge("test_gauge", 100.0);
        assert_eq!(metrics.get_gauge("test_gauge"), 100.0);
    }

    #[test]
    fn test_histogram() {
        let metrics = MetricsCollector::new();
        
        for i in 1..=100 {
            metrics.record_histogram("test_hist", i as f64);
        }
        
        let stats = metrics.get_histogram_stats("test_hist").unwrap();
        assert_eq!(stats.count, 100);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 100.0);
        assert!(stats.p50 > 45.0 && stats.p50 < 55.0);
    }

    #[test]
    fn test_timer() {
        let metrics = MetricsCollector::new();
        
        metrics.start_timer("test_timer");
        std::thread::sleep(Duration::from_millis(10));
        let duration = metrics.stop_timer("test_timer");
        
        assert!(duration.is_some());
        assert!(duration.unwrap() >= 10);
    }

    #[test]
    fn test_export_metrics() {
        let metrics = MetricsCollector::new();
        
        metrics.increment("counter1", 10);
        metrics.set_gauge("gauge1", 42.0);
        metrics.record_histogram("hist1", 100.0);
        
        let snapshot = metrics.export_metrics();
        
        assert_eq!(snapshot.counters.get("counter1"), Some(&10));
        assert_eq!(snapshot.gauges.get("gauge1"), Some(&42.0));
        assert!(snapshot.histograms.contains_key("hist1"));
    }

    #[test]
    fn test_reset() {
        let metrics = MetricsCollector::new();
        
        metrics.increment("counter1", 10);
        metrics.set_gauge("gauge1", 42.0);
        
        metrics.reset();
        
        assert_eq!(metrics.get_counter("counter1"), 0);
        assert_eq!(metrics.get_gauge("gauge1"), 0.0);
    }
}

