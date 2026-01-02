use std::collections::{HashMap, BTreeMap};
use anyhow::Result;
use serde::{Serialize, Deserialize};

use crate::block::{Block, BlockHash};
use crate::vote::Vote;
use crate::proof_of_stake::ProofOfStake;

const FINALITY_THRESHOLD: u64 = 32;
const SUPERMAJORITY_THRESHOLD: f64 = 0.66667; // 2/3

/// Tracks block confirmations and finality
pub struct FinalityTracker {
    vote_by_slot: BTreeMap<u64, HashMap<BlockHash, Vec<Vote>>>,
    finalized_slot: u64,
}

impl FinalityTracker {
    pub fn new() -> Self {
        Self {
            vote_by_slot: BTreeMap::new(),
            finalized_slot: 0,
        }
    }

    /// Add a vote for a block
    pub fn add_vote(&mut self, vote: Vote) {
        let slot_votes = self.vote_by_slot.entry(vote.slot)
            .or_insert_with(HashMap::new);
        
        slot_votes.entry(vote.block_hash.clone())
            .or_insert_with(Vec::new)
            .push(vote);
    }

    /// Calculate stake voting for a specific block
    pub fn get_stake_for_block(
        &self,
        slot: u64,
        block_hash: &BlockHash,
        pos: &ProofOfStake,
    ) -> u64 {
        let Some(slot_votes) = self.vote_by_slot.get(&slot) else {
            return 0;
        };

        let Some(votes) = slot_votes.get(block_hash) else {
            return 0;
        };

        votes.iter()
            .filter_map(|vote| {
                pos.get_validator(&vote.validator)
                    .map(|validator| validator.stake)
            })
            .sum()
    }

    /// Check if block has supermajority confirmation
    pub fn has_supermajority(
        &self,
        slot: u64,
        block_hash: &BlockHash,
        pos: &ProofOfStake,
    ) -> bool {
        let total_stake = pos.get_total_stake();
        if total_stake == 0 {
            return false;
        }

        let voting_stake = self.get_stake_for_block(slot, block_hash, pos);
        let vote_percentage = voting_stake as f64 / total_stake as f64;
        
        vote_percentage >= SUPERMAJORITY_THRESHOLD
    }

    /// Update finalized slot based on confirmations
    pub fn update_finalized_slot(&mut self, current_slot: u64, pos: &ProofOfStake) -> u64 {
        let mut new_finalized = self.finalized_slot;
        
        // Check slots from current finalized to current - threshold
        let check_until = current_slot.saturating_sub(FINALITY_THRESHOLD);
        
        for slot in (self.finalized_slot + 1)..=check_until {
            if let Some(slot_votes) = self.vote_by_slot.get(&slot) {
                // Find block with supermajority
                for (block_hash, _votes) in slot_votes {
                    if self.has_supermajority(slot, block_hash, pos) {
                        new_finalized = slot;
                        break;
                    }
                }
            }
        }
        
        self.finalized_slot = new_finalized;
        
        // Prune old votes
        self.prune_votes(self.finalized_slot.saturating_sub(100));
        
        new_finalized
    }

    /// Get finalized slot
    pub fn get_finalized_slot(&self) -> u64 {
        self.finalized_slot
    }

    /// Check if a slot is finalized
    pub fn is_slot_finalized(&self, slot: u64) -> bool {
        slot <= self.finalized_slot
    }

    /// Prune votes for old slots
    fn prune_votes(&mut self, before_slot: u64) {
        self.vote_by_slot.retain(|&slot, _| slot >= before_slot);
    }

    /// Get vote count for a slot
    pub fn get_vote_count(&self, slot: u64) -> usize {
        self.vote_by_slot.get(&slot)
            .map(|votes| votes.values().map(|v| v.len()).sum())
            .unwrap_or(0)
    }

    /// Get voting statistics
    pub fn get_stats(&self) -> FinalityStats {
        let total_slots_tracked = self.vote_by_slot.len();
        let total_votes: usize = self.vote_by_slot.values()
            .flat_map(|slot_votes| slot_votes.values())
            .map(|votes| votes.len())
            .sum();
        
        FinalityStats {
            finalized_slot: self.finalized_slot,
            total_slots_tracked,
            total_votes,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalityStats {
    pub finalized_slot: u64,
    pub total_slots_tracked: usize,
    pub total_votes: usize,
}

impl Default for FinalityTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Fork choice rule implementation
pub struct ForkChoice {
    blocks_by_slot: HashMap<u64, Vec<Block>>,
}

impl ForkChoice {
    pub fn new() -> Self {
        Self {
            blocks_by_slot: HashMap::new(),
        }
    }

    /// Add a block to fork choice
    pub fn add_block(&mut self, block: Block) {
        self.blocks_by_slot.entry(block.header.slot)
            .or_insert_with(Vec::new)
            .push(block);
    }

    /// Select best block at a slot based on stake voting
    pub fn select_best_block(
        &self,
        slot: u64,
        finality_tracker: &FinalityTracker,
        pos: &ProofOfStake,
    ) -> Option<&Block> {
        let Some(candidates) = self.blocks_by_slot.get(&slot) else {
            return None;
        };

        if candidates.is_empty() {
            return None;
        }

        // Find block with most stake voting for it
        let mut best_block = &candidates[0];
        let mut best_stake = finality_tracker.get_stake_for_block(
            slot,
            &best_block.hash(),
            pos,
        );

        for block in &candidates[1..] {
            let stake = finality_tracker.get_stake_for_block(
                slot,
                &block.hash(),
                pos,
            );
            
            if stake > best_stake {
                best_stake = stake;
                best_block = block;
            }
        }

        Some(best_block)
    }

    /// Prune blocks for old slots
    pub fn prune_before_slot(&mut self, slot: u64) {
        self.blocks_by_slot.retain(|&s, _| s >= slot);
    }
}

impl Default for ForkChoice {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof_of_stake::ValidatorInfo;

    #[test]
    fn test_finality_tracker() {
        let mut tracker = FinalityTracker::new();
        
        let vote = Vote {
            slot: 100,
            block_hash: vec![1u8; 32],
            validator: vec![2u8; 32],
            timestamp: 1234567890,
        };
        
        tracker.add_vote(vote);
        assert_eq!(tracker.get_vote_count(100), 1);
    }

    #[test]
    fn test_supermajority() {
        let mut tracker = FinalityTracker::new();
        let mut pos = ProofOfStake::new(100_000, 432_000);
        
        // Register validators
        let val1 = ValidatorInfo {
            pubkey: vec![1u8; 32],
            stake: 300_000,
            commission: 5,
            last_vote_slot: 0,
            active: true,
            activated_epoch: 0,
            deactivation_epoch: None,
        };
        
        let val2 = ValidatorInfo {
            pubkey: vec![2u8; 32],
            stake: 100_000,
            commission: 5,
            last_vote_slot: 0,
            active: true,
            activated_epoch: 0,
            deactivation_epoch: None,
        };
        
        pos.register_validator(val1).unwrap();
        pos.register_validator(val2).unwrap();
        
        let block_hash = vec![1u8; 32];
        
        // Add vote from val1 (75% of stake)
        tracker.add_vote(Vote {
            slot: 100,
            block_hash: block_hash.clone(),
            validator: vec![1u8; 32],
            timestamp: 1234567890,
        });
        
        assert!(tracker.has_supermajority(100, &block_hash, &pos));
    }

    #[test]
    fn test_finality_update() {
        let mut tracker = FinalityTracker::new();
        let mut pos = ProofOfStake::new(100_000, 432_000);
        
        let validator = ValidatorInfo {
            pubkey: vec![1u8; 32],
            stake: 1_000_000,
            commission: 5,
            last_vote_slot: 0,
            active: true,
            activated_epoch: 0,
            deactivation_epoch: None,
        };
        
        pos.register_validator(validator).unwrap();
        
        // Vote for slot 100
        let block_hash = vec![1u8; 32];
        tracker.add_vote(Vote {
            slot: 100,
            block_hash,
            validator: vec![1u8; 32],
            timestamp: 1234567890,
        });
        
        // Update with current slot = 140 (100 + 32 + buffer)
        let finalized = tracker.update_finalized_slot(140, &pos);
        assert_eq!(finalized, 100);
    }

    #[test]
    fn test_fork_choice() {
        let mut fork_choice = ForkChoice::new();
        
        let block1 = Block::new(100, vec![0u8; 32], vec![], vec![1u8; 32]);
        let block2 = Block::new(100, vec![0u8; 32], vec![], vec![2u8; 32]);
        
        fork_choice.add_block(block1);
        fork_choice.add_block(block2);
        
        let tracker = FinalityTracker::new();
        let pos = ProofOfStake::new(100_000, 432_000);
        
        let best = fork_choice.select_best_block(100, &tracker, &pos);
        assert!(best.is_some());
    }
}


