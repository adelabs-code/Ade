use serde::{Serialize, Deserialize};
use crate::block::BlockHash;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vote {
    pub slot: u64,
    pub block_hash: BlockHash,
    pub validator: Vec<u8>,
    pub timestamp: u64,
}

impl Vote {
    pub fn new(slot: u64, block_hash: BlockHash, validator: Vec<u8>) -> Self {
        Self {
            slot,
            block_hash,
            validator,
            timestamp: current_timestamp(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteState {
    pub votes: Vec<Vote>,
    pub root_slot: u64,
    pub last_timestamp: u64,
}

impl VoteState {
    pub fn new() -> Self {
        Self {
            votes: Vec::new(),
            root_slot: 0,
            last_timestamp: 0,
        }
    }

    pub fn add_vote(&mut self, vote: Vote) {
        self.last_timestamp = vote.timestamp;
        self.votes.push(vote);
        
        self.votes.retain(|v| v.slot > self.root_slot.saturating_sub(100));
    }

    pub fn update_root(&mut self, new_root: u64) {
        if new_root > self.root_slot {
            self.root_slot = new_root;
        }
    }
}

impl Default for VoteState {
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





