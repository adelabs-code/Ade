use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, debug};

use crate::bridge::Bridge;
use crate::relayer::Relayer;
use crate::proof_verification::{ProofVerifier, ProofBuilder};
use crate::event_parser::{EventParser, EventEmitter};

/// Coordinates all bridge components
pub struct BridgeCoordinator {
    bridge: Arc<Bridge>,
    relayer: Arc<RwLock<Relayer>>,
    proof_verifier: Arc<ProofVerifier>,
    event_emitter: Arc<RwLock<EventEmitter>>,
}

impl BridgeCoordinator {
    pub fn new(
        bridge: Arc<Bridge>,
        relayer: Arc<RwLock<Relayer>>,
        proof_verifier: Arc<ProofVerifier>,
        event_emitter: Arc<RwLock<EventEmitter>>,
    ) -> Self {
        Self {
            bridge,
            relayer,
            proof_verifier,
            event_emitter,
        }
    }

    /// Start the bridge coordinator
    pub async fn start(&self) -> Result<()> {
        info!("Starting bridge coordinator");

        // Start relayer
        let relayer = self.relayer.clone();
        tokio::spawn(async move {
            let relayer_guard = relayer.write().await;
            if let Err(e) = relayer_guard.start().await {
                warn!("Relayer error: {}", e);
            }
        });

        // Start event monitoring
        self.spawn_event_monitor();

        // Start proof processing
        self.spawn_proof_processor();

        Ok(())
    }

    /// Monitor for new events
    fn spawn_event_monitor(&self) {
        let event_emitter = self.event_emitter.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(5));

            loop {
                interval.tick().await;

                let emitter = event_emitter.read().await;
                let recent_events = emitter.get_events_since(0); // Get all for now
                
                debug!("Monitoring {} events", recent_events.len());
            }
        });
    }

    /// Process proofs
    fn spawn_proof_processor(&self) {
        let proof_verifier = self.proof_verifier.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(10));

            loop {
                interval.tick().await;
                debug!("Processing proofs...");
                
                // In production, process pending proofs
            }
        });
    }

    /// Handle deposit from Solana
    pub async fn handle_solana_deposit(
        &self,
        tx_hash: Vec<u8>,
        block_number: u64,
        logs: Vec<String>,
    ) -> Result<Vec<u8>> {
        info!("Handling Solana deposit, block: {}", block_number);

        // 1. Parse deposit event from logs
        let deposit_event = EventParser::parse_deposit_from_logs(&logs)?;

        // 2. Create event data
        let event_data = bincode::serialize(&deposit_event)?;

        // 3. Build proof (relayers would add signatures)
        let proof = ProofBuilder::new(tx_hash, block_number, event_data.clone())
            .build();

        // 4. Verify current block for finality
        // In production, would check against actual chain state

        // 5. Emit event on Ade side
        let mut emitter = self.event_emitter.write().await;
        emitter.emit_deposit(deposit_event, block_number)?;

        info!("Deposit handled successfully");
        Ok(event_data)
    }

    /// Handle withdrawal to Solana
    pub async fn handle_ade_withdrawal(
        &self,
        tx_hash: Vec<u8>,
        block_number: u64,
        logs: Vec<String>,
    ) -> Result<Vec<u8>> {
        info!("Handling Ade withdrawal, block: {}", block_number);

        // 1. Parse withdrawal event
        let withdrawal_event = EventParser::parse_withdrawal_from_logs(&logs)?;

        // 2. Create event data
        let event_data = bincode::serialize(&withdrawal_event)?;

        // 3. Build proof
        let proof = ProofBuilder::new(tx_hash, block_number, event_data.clone())
            .build();

        // 4. Emit event
        let mut emitter = self.event_emitter.write().await;
        emitter.emit_withdrawal(withdrawal_event, block_number)?;

        info!("Withdrawal handled successfully");
        Ok(event_data)
    }

    /// Process a verified proof
    pub async fn process_verified_proof(
        &self,
        proof: crate::solana_lock::BridgeProof,
        current_block: u64,
    ) -> Result<()> {
        info!("Processing verified proof");

        // Verify proof
        let verification = self.proof_verifier.verify_proof(&proof, current_block)?;

        if !verification.valid {
            warn!("Proof verification failed: {:?}", verification.errors);
            return Err(anyhow::anyhow!("Invalid proof"));
        }

        info!("Proof verified: {} relayers, merkle: {}, finality: {}",
            verification.relayers_verified,
            verification.merkle_verified,
            verification.finality_verified
        );

        // Process based on event type
        // Parse event_data and execute mint/unlock

        Ok(())
    }

    /// Get coordinator statistics
    pub async fn get_stats(&self) -> CoordinatorStats {
        let relayer = self.relayer.read().await;
        let emitter = self.event_emitter.read().await;

        CoordinatorStats {
            relayer_stats: relayer.get_stats(),
            bridge_stats: self.bridge.get_stats(),
            total_events: emitter.get_events_since(0).len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CoordinatorStats {
    pub relayer_stats: crate::relayer::RelayerStats,
    pub bridge_stats: crate::bridge::BridgeStats,
    pub total_events: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::BridgeConfig;

    #[test]
    fn test_event_parsing() {
        let logs = vec![
            "Program log: Starting".to_string(),
            "DEPOSIT_EVENT: {...}".to_string(),
        ];

        let result = EventParser::parse_deposit_from_logs(&logs);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_coordinator_creation() {
        let config = BridgeConfig {
            solana_rpc_url: "http://localhost:8899".to_string(),
            sidechain_rpc_url: "http://localhost:8899".to_string(),
            bridge_program_id: vec![1; 32],
            supported_tokens: vec![],
            min_confirmations: 32,
            relayer_set: crate::bridge::RelayerSet {
                relayers: vec![vec![1; 32]],
                threshold: 1,
            },
            fraud_proof_window: 86400,
        };

        let bridge = Arc::new(Bridge::new(config));
        let relayer_config = crate::relayer::RelayerConfig {
            poll_interval_ms: 1000,
            max_retry_attempts: 3,
            confirmation_threshold: 32,
            batch_size: 10,
            max_concurrent_relays: 5,
            solana_rpc_url: "http://localhost:8899".to_string(),
            ade_rpc_url: "http://localhost:8899".to_string(),
        };
        let relayer = Arc::new(RwLock::new(Relayer::new(relayer_config)));
        let verifier = Arc::new(ProofVerifier::new(vec![vec![1; 32]], 1, 32));
        let emitter = Arc::new(RwLock::new(EventEmitter::new()));

        let coordinator = BridgeCoordinator::new(bridge, relayer, verifier, emitter);
        
        // Should not panic
    }
}



