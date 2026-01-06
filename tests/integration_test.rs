use ade_node::*;
use ade_transaction::*;
use ade_consensus::*;
use ed25519_dalek::Keypair;
use rand::rngs::OsRng;
use std::sync::Arc;
use tempfile::TempDir;

#[tokio::test]
async fn test_full_transaction_lifecycle() {
    // Setup
    let temp_dir = TempDir::new().unwrap();
    let storage = Arc::new(Storage::new(temp_dir.path().to_str().unwrap()).unwrap());
    let accounts = Arc::new(std::sync::RwLock::new(std::collections::HashMap::new()));
    
    // Create accounts
    let mut csprng = OsRng;
    let sender_keypair = Keypair::generate(&mut csprng);
    let sender_pubkey = sender_keypair.public.to_bytes().to_vec();
    
    let recipient_pubkey = vec![2u8; 32];
    
    // Fund sender account
    {
        let mut accounts_map = accounts.write().unwrap();
        accounts_map.insert(sender_pubkey.clone(), Account::new(1_000_000_000, vec![]));
    }
    
    // Create transfer transaction
    let transfer_ix = Instruction::transfer(
        sender_pubkey.clone(),
        recipient_pubkey.clone(),
        500_000_000,
        vec![0u8; 32],
    );
    
    let tx = Transaction::new(
        &[&sender_keypair],
        vec![transfer_ix],
        vec![1u8; 32],
    ).unwrap();
    
    // Verify transaction
    assert!(tx.verify().is_ok());
    
    // Execute transaction
    let executor = TransactionExecutor::new(accounts.clone());
    let result = executor.execute_transaction(&tx, 1).unwrap();
    
    assert!(result.success);
    assert_eq!(result.modified_accounts.len(), 2);
    
    // Verify balances
    let accounts_map = accounts.read().unwrap();
    let sender_account = accounts_map.get(&sender_pubkey).unwrap();
    let recipient_account = accounts_map.get(&recipient_pubkey).unwrap();
    
    assert_eq!(sender_account.lamports, 500_000_000);
    assert_eq!(recipient_account.lamports, 500_000_000);
}

#[tokio::test]
async fn test_block_production_flow() {
    let temp_dir = TempDir::new().unwrap();
    let storage = Arc::new(Storage::new(temp_dir.path().to_str().unwrap()).unwrap());
    
    let mut csprng = OsRng;
    let validator_keypair = Keypair::generate(&mut csprng);
    
    let mempool = Arc::new(Mempool::new(MempoolConfig::default()));
    let fee_market = Arc::new(FeeMarket::new(FeeMarketConfig::default()));
    
    // Add transactions to mempool
    for _ in 0..5 {
        let keypair = Keypair::generate(&mut csprng);
        let tx = Transaction::new(&[&keypair], vec![], vec![1u8; 32]).unwrap();
        mempool.add_transaction(tx, 10_000, 1_000).unwrap();
    }
    
    assert_eq!(mempool.size(), 5);
    
    // Produce block
    let producer = BlockProducer::new(
        validator_keypair,
        mempool.clone(),
        fee_market,
        storage,
        ProducerConfig::default(),
    );
    
    let result = producer.produce_block(1, vec![0u8; 32]).await.unwrap();
    
    assert_eq!(result.transaction_count, 5);
    assert_eq!(mempool.size(), 0); // Transactions removed from mempool
}

#[tokio::test]
async fn test_consensus_finality() {
    let mut pos = ProofOfStake::new(100_000, 432_000);
    
    // Register validators
    let val1 = ValidatorInfo {
        pubkey: vec![1u8; 32],
        stake: 400_000,
        commission: 5,
        last_vote_slot: 0,
        active: true,
        activated_epoch: 0,
        deactivation_epoch: None,
    };
    
    let val2 = ValidatorInfo {
        pubkey: vec![2u8; 32],
        stake: 300_000,
        commission: 5,
        last_vote_slot: 0,
        active: true,
        activated_epoch: 0,
        deactivation_epoch: None,
    };
    
    pos.register_validator(val1).unwrap();
    pos.register_validator(val2).unwrap();
    
    // Create finality tracker
    let mut finality_tracker = FinalityTracker::new();
    
    // Add votes for slot 100
    let block_hash = vec![1u8; 32];
    
    finality_tracker.add_vote(Vote {
        slot: 100,
        block_hash: block_hash.clone(),
        validator: vec![1u8; 32],
        timestamp: 12345,
    });
    
    finality_tracker.add_vote(Vote {
        slot: 100,
        block_hash: block_hash.clone(),
        validator: vec![2u8; 32],
        timestamp: 12345,
    });
    
    // Check supermajority (100% of stake voted)
    assert!(finality_tracker.has_supermajority(100, &block_hash, &pos));
    
    // Update finalized slot
    let finalized = finality_tracker.update_finalized_slot(140, &pos);
    assert_eq!(finalized, 100);
}

#[tokio::test]
async fn test_ai_agent_deployment_and_execution() {
    let runtime = AIRuntime::new(10, 30000);
    
    let agent_id = vec![1u8; 32];
    let model_hash = vec![2u8; 32];
    let owner = vec![3u8; 32];
    
    let config = AgentConfig {
        model_type: "transformer".to_string(),
        parameters: std::collections::HashMap::new(),
        max_execution_time: 30000,
        allowed_operations: vec!["inference".to_string()],
        compute_budget: 100000,
    };
    
    // Deploy agent
    runtime.deploy_agent(agent_id.clone(), model_hash, owner.clone(), config).unwrap();
    
    // Verify deployment
    let info = runtime.get_agent_info(&agent_id).unwrap();
    assert_eq!(info.execution_count, 0);
    
    // Execute agent
    let request = ExecutionRequest {
        agent_id: agent_id.clone(),
        input_data: vec![1, 2, 3, 4],
        max_compute: 50000,
        caller: owner,
    };
    
    let result = runtime.execute_agent(request).await.unwrap();
    assert!(result.success);
    
    // Verify execution count updated
    let info = runtime.get_agent_info(&agent_id).unwrap();
    assert_eq!(info.execution_count, 1);
}

#[test]
fn test_merkle_proof_verification() {
    use ade_bridge::MerkleTree;
    
    let leaves = vec![
        vec![1u8; 32],
        vec![2u8; 32],
        vec![3u8; 32],
        vec![4u8; 32],
    ];
    
    let tree = MerkleTree::new(leaves).unwrap();
    
    // Generate and verify proof for each leaf
    for i in 0..4 {
        let proof = tree.generate_proof(i).unwrap();
        assert!(MerkleTree::verify_proof(&proof));
    }
}

#[tokio::test]
async fn test_mempool_fee_market_integration() {
    let mempool = Mempool::new(MempoolConfig::default());
    let fee_market = FeeMarket::new(FeeMarketConfig::default());
    
    let mut csprng = OsRng;
    
    // Add transactions with different fees
    for i in 0..10 {
        let keypair = Keypair::generate(&mut csprng);
        let tx = Transaction::new(&[&keypair], vec![], vec![1u8; 32]).unwrap();
        let fee = 10_000 + (i * 1_000);
        mempool.add_transaction(tx, fee, 0).unwrap();
    }
    
    // Get top transactions
    let top_txs = mempool.get_top_transactions(5);
    assert_eq!(top_txs.len(), 5);
    
    // Verify ordering (highest fee first)
    for i in 0..top_txs.len()-1 {
        assert!(top_txs[i].fee >= top_txs[i+1].fee);
    }
    
    // Record block in fee market
    fee_market.record_block(1, 5, 10, vec![1000, 2000, 3000]);
    
    let estimate = fee_market.estimate_fees();
    assert!(estimate.medium_fee > estimate.low_fee);
}

#[tokio::test]
async fn test_network_message_protocol() {
    use ade_node::Protocol;
    
    let message = GossipMessage::Ping { timestamp: 12345 };
    
    // Encode
    let encoded = Protocol::encode(&message).unwrap();
    
    // Decode
    let decoded = Protocol::decode(&encoded).unwrap();
    
    // Verify
    match decoded {
        GossipMessage::Ping { timestamp } => assert_eq!(timestamp, 12345),
        _ => panic!("Wrong message type"),
    }
}

#[test]
fn test_compute_metering() {
    let mut meter = ComputeMeter::new(100_000);
    
    // Consume various operations
    meter.consume("sha256").unwrap();
    meter.consume("ed25519_verify").unwrap();
    meter.consume("model_load").unwrap();
    
    let stats = meter.get_stats();
    assert!(stats.consumed > 0);
    assert!(stats.consumed < stats.budget);
    assert_eq!(stats.operation_count, 3);
}

#[tokio::test]
async fn test_end_to_end_block_flow() {
    // This test simulates the complete flow:
    // Transaction -> Mempool -> Block Production -> State Transition -> Finality
    
    let temp_dir = TempDir::new().unwrap();
    let storage = Arc::new(Storage::new(temp_dir.path().to_str().unwrap()).unwrap());
    let accounts = Arc::new(std::sync::RwLock::new(std::collections::HashMap::new()));
    
    // Setup
    let mut csprng = OsRng;
    let validator_keypair = Keypair::generate(&mut csprng);
    let sender_keypair = Keypair::generate(&mut csprng);
    
    // Fund sender
    {
        let mut accounts_map = accounts.write().unwrap();
        accounts_map.insert(
            sender_keypair.public.to_bytes().to_vec(),
            Account::new(1_000_000_000, vec![])
        );
    }
    
    // Create transaction
    let transfer_ix = Instruction::transfer(
        sender_keypair.public.to_bytes().to_vec(),
        vec![2u8; 32],
        100_000_000,
        vec![0u8; 32],
    );
    
    let tx = Transaction::new(&[&sender_keypair], vec![transfer_ix], vec![1u8; 32]).unwrap();
    
    // Add to mempool
    let mempool = Arc::new(Mempool::new(MempoolConfig::default()));
    mempool.add_transaction(tx, 10_000, 1_000).unwrap();
    
    // Produce block
    let fee_market = Arc::new(FeeMarket::new(FeeMarketConfig::default()));
    let producer = BlockProducer::new(
        validator_keypair,
        mempool.clone(),
        fee_market,
        storage.clone(),
        ProducerConfig::default(),
    );
    
    let block_result = producer.produce_block(1, vec![0u8; 32]).await.unwrap();
    assert_eq!(block_result.transaction_count, 1);
    
    // Apply state transition
    let transition = StateTransition::new(storage.clone());
    let transition_result = transition.apply_block(&block_result.block).await.unwrap();
    
    assert_eq!(transition_result.successful_transactions, 1);
    assert!(transition_result.account_changes.len() > 0);
}






