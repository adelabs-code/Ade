use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use borsh::{BorshDeserialize, BorshSerialize};

/// Parse Solana program logs for bridge events
pub struct EventParser;

impl EventParser {
    /// Parse deposit event from Solana logs
    pub fn parse_deposit_from_logs(logs: &[String]) -> Result<ParsedDepositEvent> {
        for log in logs {
            if log.starts_with("DEPOSIT_EVENT:") {
                let event_str = log.strip_prefix("DEPOSIT_EVENT:").unwrap().trim();
                
                // In production, would use proper deserialization
                // For now, extract key information
                return Self::parse_deposit_log(event_str);
            }
        }

        Err(anyhow::anyhow!("No deposit event found in logs"))
    }

    /// Parse deposit event from log string
    /// Expected formats:
    /// - "amount=1000000,token=So11...1111,depositor=ABC...XYZ,recipient=DEF...UVW,nonce=42"
    /// - JSON: {"amount":1000000,"token":"So11...","depositor":"ABC...","recipient":"DEF...","nonce":42}
    fn parse_deposit_log(log: &str) -> Result<ParsedDepositEvent> {
        // Try JSON parsing first
        if log.starts_with('{') {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(log) {
                return Ok(ParsedDepositEvent {
                    depositor: Self::parse_pubkey_from_json(&parsed, "depositor"),
                    token: Self::parse_pubkey_from_json(&parsed, "token"),
                    amount: parsed.get("amount")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0),
                    target_chain: parsed.get("target_chain")
                        .or(parsed.get("targetChain"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("ade")
                        .to_string(),
                    recipient: Self::parse_pubkey_from_json(&parsed, "recipient"),
                    nonce: parsed.get("nonce")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0),
                });
            }
        }
        
        // Try key=value parsing
        let mut depositor = vec![];
        let mut token = vec![];
        let mut amount = 0u64;
        let mut target_chain = String::from("ade");
        let mut recipient = vec![];
        let mut nonce = 0u64;
        
        for pair in log.split(',') {
            let parts: Vec<&str> = pair.trim().split('=').collect();
            if parts.len() != 2 {
                continue;
            }
            
            let key = parts[0].trim().to_lowercase();
            let value = parts[1].trim();
            
            match key.as_str() {
                "depositor" | "sender" => {
                    depositor = bs58::decode(value).into_vec().unwrap_or_default();
                }
                "token" | "mint" => {
                    token = bs58::decode(value).into_vec().unwrap_or_default();
                }
                "amount" => {
                    amount = value.parse().unwrap_or(0);
                }
                "target_chain" | "targetchain" | "to" => {
                    target_chain = value.to_string();
                }
                "recipient" | "receiver" => {
                    recipient = bs58::decode(value).into_vec().unwrap_or_default();
                }
                "nonce" => {
                    nonce = value.parse().unwrap_or(0);
                }
                _ => {}
            }
        }
        
        Ok(ParsedDepositEvent {
            depositor,
            token,
            amount,
            target_chain,
            recipient,
            nonce,
        })
    }
    
    /// Helper to parse a pubkey from JSON value
    fn parse_pubkey_from_json(json: &serde_json::Value, key: &str) -> Vec<u8> {
        json.get(key)
            .and_then(|v| v.as_str())
            .and_then(|s| bs58::decode(s).into_vec().ok())
            .unwrap_or_default()
    }

    /// Parse withdrawal event from Ade logs
    pub fn parse_withdrawal_from_logs(logs: &[String]) -> Result<ParsedWithdrawalEvent> {
        for log in logs {
            if log.starts_with("BURN_EVENT:") {
                let event_str = log.strip_prefix("BURN_EVENT:").unwrap().trim();
                return Self::parse_withdrawal_log(event_str);
            }
        }

        Err(anyhow::anyhow!("No withdrawal event found in logs"))
    }

    /// Parse withdrawal event from log string
    fn parse_withdrawal_log(log: &str) -> Result<ParsedWithdrawalEvent> {
        // Try JSON parsing first
        if log.starts_with('{') {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(log) {
                return Ok(ParsedWithdrawalEvent {
                    burner: Self::parse_pubkey_from_json(&parsed, "burner"),
                    token: Self::parse_pubkey_from_json(&parsed, "token"),
                    amount: parsed.get("amount")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0),
                    target_chain: parsed.get("target_chain")
                        .or(parsed.get("targetChain"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("solana")
                        .to_string(),
                    recipient: Self::parse_pubkey_from_json(&parsed, "recipient"),
                    nonce: parsed.get("nonce")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0),
                });
            }
        }
        
        // Try key=value parsing
        let mut burner = vec![];
        let mut token = vec![];
        let mut amount = 0u64;
        let mut target_chain = String::from("solana");
        let mut recipient = vec![];
        let mut nonce = 0u64;
        
        for pair in log.split(',') {
            let parts: Vec<&str> = pair.trim().split('=').collect();
            if parts.len() != 2 {
                continue;
            }
            
            let key = parts[0].trim().to_lowercase();
            let value = parts[1].trim();
            
            match key.as_str() {
                "burner" | "sender" => {
                    burner = bs58::decode(value).into_vec().unwrap_or_default();
                }
                "token" | "mint" => {
                    token = bs58::decode(value).into_vec().unwrap_or_default();
                }
                "amount" => {
                    amount = value.parse().unwrap_or(0);
                }
                "target_chain" | "targetchain" | "to" => {
                    target_chain = value.to_string();
                }
                "recipient" | "receiver" => {
                    recipient = bs58::decode(value).into_vec().unwrap_or_default();
                }
                "nonce" => {
                    nonce = value.parse().unwrap_or(0);
                }
                _ => {}
            }
        }
        
        Ok(ParsedWithdrawalEvent {
            burner,
            token,
            amount,
            target_chain,
            recipient,
            nonce,
        })
    }

    /// Extract event data from transaction bytes
    /// Parses the transaction structure to find bridge-related events
    pub fn extract_event_from_transaction(tx_data: &[u8]) -> Result<Vec<u8>> {
        if tx_data.is_empty() {
            return Err(anyhow::anyhow!("Empty transaction data"));
        }
        
        // Try to deserialize as JSON first (for RPC responses)
        if let Ok(tx_str) = std::str::from_utf8(tx_data) {
            if tx_str.starts_with('{') {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(tx_str) {
                    // Extract logs from transaction metadata
                    if let Some(logs) = parsed.get("meta")
                        .and_then(|m| m.get("logMessages"))
                        .and_then(|l| l.as_array())
                    {
                        for log in logs {
                            if let Some(log_str) = log.as_str() {
                                if log_str.contains("DepositEvent") || log_str.contains("BridgeDeposit") {
                                    return Ok(log_str.as_bytes().to_vec());
                                }
                                if log_str.contains("WithdrawEvent") || log_str.contains("BridgeBurn") {
                                    return Ok(log_str.as_bytes().to_vec());
                                }
                            }
                        }
                    }
                    
                    // Try instruction data if no logs found
                    if let Some(instructions) = parsed.get("transaction")
                        .and_then(|t| t.get("message"))
                        .and_then(|m| m.get("instructions"))
                        .and_then(|i| i.as_array())
                    {
                        for instruction in instructions {
                            if let Some(data) = instruction.get("data").and_then(|d| d.as_str()) {
                                if let Ok(decoded) = bs58::decode(data).into_vec() {
                                    // Check for bridge instruction discriminator (first byte)
                                    if !decoded.is_empty() && (decoded[0] == 1 || decoded[0] == 2) {
                                        return Ok(decoded);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Handle binary transaction format
        // Check if it looks like a Solana transaction (signature + message)
        if tx_data.len() > 64 {
            // Skip signature (64 bytes) and try to parse message
            let message_data = &tx_data[64..];
            
            // Look for bridge instruction data in the message
            // This is a simplified parser - in production, use proper Solana SDK deserialization
            if message_data.len() > 10 {
                return Ok(message_data.to_vec());
            }
        }

        Err(anyhow::anyhow!("No bridge event found in transaction"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedDepositEvent {
    pub depositor: Vec<u8>,
    pub token: Vec<u8>,
    pub amount: u64,
    pub target_chain: String,
    pub recipient: Vec<u8>,
    pub nonce: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedWithdrawalEvent {
    pub burner: Vec<u8>,
    pub token: Vec<u8>,
    pub amount: u64,
    pub target_chain: String,
    pub recipient: Vec<u8>,
    pub nonce: u64,
}

/// Event emitter for Ade sidechain
pub struct EventEmitter {
    events: Vec<EmittedEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmittedEvent {
    pub event_type: EventType,
    pub data: Vec<u8>,
    pub slot: u64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    Deposit,
    Withdrawal,
    Mint,
    Burn,
}

impl EventEmitter {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
        }
    }

    /// Emit deposit event
    pub fn emit_deposit(&mut self, event: ParsedDepositEvent, slot: u64) -> Result<()> {
        let data = bincode::serialize(&event)?;
        
        self.events.push(EmittedEvent {
            event_type: EventType::Deposit,
            data,
            slot,
            timestamp: current_timestamp(),
        });

        Ok(())
    }

    /// Emit withdrawal event
    pub fn emit_withdrawal(&mut self, event: ParsedWithdrawalEvent, slot: u64) -> Result<()> {
        let data = bincode::serialize(&event)?;
        
        self.events.push(EmittedEvent {
            event_type: EventType::Withdrawal,
            data,
            slot,
            timestamp: current_timestamp(),
        });

        Ok(())
    }

    /// Get events since slot
    pub fn get_events_since(&self, slot: u64) -> Vec<&EmittedEvent> {
        self.events.iter()
            .filter(|e| e.slot >= slot)
            .collect()
    }

    /// Get events by type
    pub fn get_events_by_type(&self, event_type: EventType) -> Vec<&EmittedEvent> {
        self.events.iter()
            .filter(|e| matches!(e.event_type, event_type))
            .collect()
    }

    /// Clear old events
    pub fn prune_events(&mut self, before_slot: u64) -> usize {
        let initial_len = self.events.len();
        self.events.retain(|e| e.slot >= before_slot);
        initial_len - self.events.len()
    }
}

impl Default for EventEmitter {
    fn default() -> Self {
        Self::new()
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
    fn test_event_emitter() {
        let mut emitter = EventEmitter::new();

        let deposit = ParsedDepositEvent {
            depositor: vec![1; 32],
            token: vec![2; 32],
            amount: 1000,
            target_chain: "ade".to_string(),
            recipient: vec![3; 32],
            nonce: 1,
        };

        emitter.emit_deposit(deposit, 100).unwrap();

        let events = emitter.get_events_since(50);
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_event_pruning() {
        let mut emitter = EventEmitter::new();

        let deposit = ParsedDepositEvent {
            depositor: vec![1; 32],
            token: vec![2; 32],
            amount: 1000,
            target_chain: "ade".to_string(),
            recipient: vec![3; 32],
            nonce: 1,
        };

        emitter.emit_deposit(deposit.clone(), 100).unwrap();
        emitter.emit_deposit(deposit.clone(), 200).unwrap();
        emitter.emit_deposit(deposit, 300).unwrap();

        let pruned = emitter.prune_events(250);
        assert_eq!(pruned, 2);
        assert_eq!(emitter.events.len(), 1);
    }
}

