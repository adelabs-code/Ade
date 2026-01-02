use ade_node::*;
use ade_transaction::*;
use ed25519_dalek::Keypair;
use rand::rngs::OsRng;
use std::sync::Arc;
use std::time::Instant;
use tempfile::TempDir;

/// Load test configuration
pub struct LoadTestConfig {
    pub transaction_count: usize,
    pub concurrent_threads: usize,
    pub batch_size: usize,
}

impl Default for LoadTestConfig {
    fn default() -> Self {
        Self {
            transaction_count: 1000,
            concurrent_threads: 4,
            batch_size: 100,
        }
    }
}

/// Load test results
#[derive(Debug)]
pub struct LoadTestResults {
    pub total_transactions: usize,
    pub successful_transactions: usize,
    pub failed_transactions: usize,
    pub total_time_ms: u64,
    pub tps: f64,
    pub average_latency_ms: f64,
    pub p95_latency_ms: f64,
}

#[tokio::test]
async fn test_mempool_throughput() {
    let config = LoadTestConfig::default();
    let mempool = Arc::new(Mempool::new(MempoolConfig::default()));
    
    let mut csprng = OsRng;
    let start = Instant::now();
    let mut successful = 0;
    let mut failed = 0;
    
    for _ in 0..config.transaction_count {
        let keypair = Keypair::generate(&mut csprng);
        let tx = Transaction::new(&[&keypair], vec![], vec![1u8; 32]).unwrap();
        
        match mempool.add_transaction(tx, 10_000, 1_000) {
            Ok(_) => successful += 1,
            Err(_) => failed += 1,
        }
    }
    
    let duration = start.elapsed();
    let tps = successful as f64 / duration.as_secs_f64();
    
    println!("\n=== Mempool Throughput Test ===");
    println!("Total transactions: {}", config.transaction_count);
    println!("Successful: {}", successful);
    println!("Failed: {}", failed);
    println!("Duration: {:?}", duration);
    println!("TPS: {:.2}", tps);
    
    assert!(tps > 1000.0); // Should handle >1000 TPS
}

#[tokio::test]
async fn test_block_production_performance() {
    let temp_dir = TempDir::new().unwrap();
    let storage = Arc::new(Storage::new(temp_dir.path().to_str().unwrap()).unwrap());
    
    let mut csprng = OsRng;
    let validator_keypair = Keypair::generate(&mut csprng);
    
    let mempool = Arc::new(Mempool::new(MempoolConfig::default()));
    let fee_market = Arc::new(FeeMarket::new(FeeMarketConfig::default()));
    
    // Fill mempool with transactions
    for _ in 0..1000 {
        let keypair = Keypair::generate(&mut csprng);
        let tx = Transaction::new(&[&keypair], vec![], vec![1u8; 32]).unwrap();
        mempool.add_transaction(tx, 10_000, 0).unwrap();
    }
    
    let producer = BlockProducer::new(
        validator_keypair,
        mempool,
        fee_market,
        storage,
        ProducerConfig::default(),
    );
    
    let start = Instant::now();
    let result = producer.produce_block(1, vec![0u8; 32]).await.unwrap();
    let duration = start.elapsed();
    
    println!("\n=== Block Production Performance ===");
    println!("Transactions packed: {}", result.transaction_count);
    println!("Production time: {:?}", duration);
    println!("Rejected: {}", result.rejected_count);
    
    assert!(duration.as_millis() < 400); // Should complete within slot time
}

#[tokio::test]
async fn test_concurrent_transaction_execution() {
    let temp_dir = TempDir::new().unwrap();
    let storage = Arc::new(Storage::new(temp_dir.path().to_str().unwrap()).unwrap());
    let accounts = Arc::new(std::sync::RwLock::new(std::collections::HashMap::new()));
    
    // Create executor
    let executor = Arc::new(TransactionExecutor::new(accounts.clone()));
    
    let mut csprng = OsRng;
    let mut handles = vec![];
    
    // Spawn concurrent execution tasks
    for i in 0..10 {
        let executor = executor.clone();
        let keypair = Keypair::generate(&mut csprng);
        
        let handle = tokio::spawn(async move {
            let tx = Transaction::new(&[&keypair], vec![], vec![1u8; 32]).unwrap();
            executor.execute_transaction(&tx, i as u64)
        });
        
        handles.push(handle);
    }
    
    // Wait for all
    let start = Instant::now();
    let results: Vec<_> = futures::future::join_all(handles).await;
    let duration = start.elapsed();
    
    let successful = results.iter().filter(|r| r.is_ok()).count();
    
    println!("\n=== Concurrent Execution Test ===");
    println!("Total tasks: {}", results.len());
    println!("Successful: {}", successful);
    println!("Duration: {:?}", duration);
}

#[tokio::test]
async fn test_storage_throughput() {
    let temp_dir = TempDir::new().unwrap();
    let storage = Storage::new(temp_dir.path().to_str().unwrap()).unwrap();
    
    let start = Instant::now();
    let count = 10_000;
    
    // Batch account storage
    let mut accounts = Vec::new();
    for i in 0..count {
        let address = vec![i as u8; 32];
        let data = vec![i as u8; 100];
        accounts.push((address, data));
    }
    
    storage.store_accounts_batch(&accounts).unwrap();
    
    let duration = start.elapsed();
    let ops_per_sec = count as f64 / duration.as_secs_f64();
    
    println!("\n=== Storage Throughput Test ===");
    println!("Accounts stored: {}", count);
    println!("Duration: {:?}", duration);
    println!("Ops/sec: {:.2}", ops_per_sec);
    
    assert!(ops_per_sec > 5000.0); // Should handle >5000 ops/sec
}

#[tokio::test]
async fn test_ai_agent_execution_throughput() {
    let runtime = AIRuntime::new(100, 30000);
    let mut csprng = OsRng;
    
    // Deploy agent
    let agent_id = vec![1u8; 32];
    let owner = vec![3u8; 32];
    let config = AgentConfig {
        model_type: "transformer".to_string(),
        parameters: std::collections::HashMap::new(),
        max_execution_time: 30000,
        allowed_operations: vec!["inference".to_string()],
        compute_budget: 100000,
    };
    
    runtime.deploy_agent(agent_id.clone(), vec![2u8; 32], owner.clone(), config).unwrap();
    
    // Execute multiple times
    let start = Instant::now();
    let count = 100;
    
    for i in 0..count {
        let request = ExecutionRequest {
            agent_id: agent_id.clone(),
            input_data: vec![i as u8; 100],
            max_compute: 50000,
            caller: owner.clone(),
        };
        
        runtime.execute_agent(request).await.unwrap();
    }
    
    let duration = start.elapsed();
    let executions_per_sec = count as f64 / duration.as_secs_f64();
    
    println!("\n=== AI Agent Execution Throughput ===");
    println!("Executions: {}", count);
    println!("Duration: {:?}", duration);
    println!("Exec/sec: {:.2}", executions_per_sec);
}

#[tokio::test]
async fn test_network_broadcast_performance() {
    let network = NetworkManager::new(9900, vec![]).unwrap();
    
    // Add peers
    for i in 0..100 {
        let address = format!("127.0.0.1:{}", 10000 + i);
        // network.connect_to_peer(&address).await.unwrap();
    }
    
    let start = Instant::now();
    let message_count = 1000;
    
    for _ in 0..message_count {
        let message = GossipMessage::Ping { timestamp: current_timestamp() };
        let _ = network.broadcast(message).await;
    }
    
    let duration = start.elapsed();
    let messages_per_sec = message_count as f64 / duration.as_secs_f64();
    
    println!("\n=== Network Broadcast Performance ===");
    println!("Messages: {}", message_count);
    println!("Duration: {:?}", duration);
    println!("Msg/sec: {:.2}", messages_per_sec);
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

