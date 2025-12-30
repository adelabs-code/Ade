# Consensus Mechanism

## Overview

Ade uses a Proof-of-Stake (PoS) consensus mechanism optimized for high throughput and low latency. The system combines stake-weighted leader selection with rapid finality.

## Core Concepts

### Validators

Validators are nodes that participate in consensus by:
- Producing blocks when selected as leader
- Voting on proposed blocks
- Maintaining network state

**Requirements:**
- Minimum stake: 100,000 ADE tokens
- Uptime: >95%
- Hardware: See node-setup.md

### Stake

Stake represents voting power in the network.

```rust
pub struct ValidatorInfo {
    pub pubkey: Vec<u8>,
    pub stake: u64,
    pub commission: u8,
    pub last_vote_slot: u64,
    pub active: bool,
}
```

### Slots and Epochs

**Slot**: 400ms time unit for block production

**Epoch**: 432,000 slots (~2 days)

```rust
pub const SLOT_DURATION_MS: u64 = 400;
pub const SLOTS_PER_EPOCH: u64 = 432_000;
```

## Leader Selection

### Algorithm

Leaders are selected deterministically based on stake weight:

```rust
pub fn select_leader(slot: u64, validators: &[ValidatorInfo]) -> Vec<u8> {
    let active_validators: Vec<_> = validators
        .iter()
        .filter(|v| v.active)
        .collect();
    
    let total_stake: u64 = active_validators
        .iter()
        .map(|v| v.stake)
        .sum();
    
    // Deterministic pseudo-random selection weighted by stake
    let seed = hash_slot(slot);
    let target = u64::from_le_bytes(seed[0..8].try_into().unwrap()) % total_stake;
    
    let mut cumulative = 0u64;
    for validator in active_validators {
        cumulative += validator.stake;
        if cumulative > target {
            return validator.pubkey.clone();
        }
    }
    
    // Fallback (should never reach)
    active_validators[0].pubkey.clone()
}
```

### Schedule

Leader schedule is computed at the start of each epoch:

```rust
pub fn compute_leader_schedule(
    epoch: u64,
    validators: &[ValidatorInfo],
) -> Vec<Vec<u8>> {
    let mut schedule = Vec::with_capacity(SLOTS_PER_EPOCH as usize);
    
    for slot in 0..SLOTS_PER_EPOCH {
        let absolute_slot = epoch * SLOTS_PER_EPOCH + slot;
        let leader = select_leader(absolute_slot, validators);
        schedule.push(leader);
    }
    
    schedule
}
```

## Block Production

### Leader Responsibilities

1. Collect transactions from mempool
2. Validate transactions
3. Build block
4. Sign block
5. Broadcast to network

```rust
pub async fn produce_block(
    slot: u64,
    parent_hash: Vec<u8>,
    validator: &Keypair,
) -> Result<Block> {
    // 1. Collect transactions
    let txs = collect_transactions(MAX_TXS_PER_BLOCK).await?;
    
    // 2. Validate
    let valid_txs: Vec<_> = txs
        .into_iter()
        .filter(|tx| validate_transaction(tx).is_ok())
        .collect();
    
    // 3. Build block
    let block = Block::new(
        slot,
        parent_hash,
        valid_txs,
        validator.public.to_bytes().to_vec(),
    );
    
    // 4. Sign
    let signature = validator.sign(&block.hash());
    
    // 5. Broadcast
    broadcast_block(&block).await?;
    
    Ok(block)
}
```

### Block Structure

```rust
pub struct Block {
    pub header: BlockHeader,
    pub transactions: Vec<Transaction>,
}

pub struct BlockHeader {
    pub slot: u64,
    pub parent_hash: Vec<u8>,
    pub transactions_root: Vec<u8>,
    pub timestamp: u64,
    pub validator: Vec<u8>,
}
```

## Voting

### Vote Transaction

Validators vote on blocks to signal agreement:

```rust
pub struct Vote {
    pub slot: u64,
    pub block_hash: Vec<u8>,
    pub validator: Vec<u8>,
    pub timestamp: u64,
}

pub fn create_vote(
    slot: u64,
    block_hash: Vec<u8>,
    validator: &Keypair,
) -> Vote {
    Vote {
        slot,
        block_hash,
        validator: validator.public.to_bytes().to_vec(),
        timestamp: current_timestamp(),
    }
}
```

### Vote Collection

```rust
pub struct VoteTracker {
    votes: HashMap<u64, Vec<Vote>>,
}

impl VoteTracker {
    pub fn add_vote(&mut self, vote: Vote) {
        self.votes
            .entry(vote.slot)
            .or_insert_with(Vec::new)
            .push(vote);
    }
    
    pub fn get_stake_voting_for(
        &self,
        slot: u64,
        block_hash: &[u8],
        validators: &[ValidatorInfo],
    ) -> u64 {
        let votes = self.votes.get(&slot).unwrap_or(&vec![]);
        
        votes
            .iter()
            .filter(|v| v.block_hash == block_hash)
            .filter_map(|v| {
                validators
                    .iter()
                    .find(|val| val.pubkey == v.validator)
                    .map(|val| val.stake)
            })
            .sum()
    }
}
```

## Finality

### Confirmation Thresholds

```rust
pub const CONFIRMATIONS_FOR_FINALITY: u64 = 32;

pub fn is_finalized(
    slot: u64,
    current_slot: u64,
) -> bool {
    current_slot >= slot + CONFIRMATIONS_FOR_FINALITY
}
```

### Supermajority Voting

A block is considered confirmed when it receives votes from >2/3 of stake:

```rust
pub fn is_confirmed(
    slot: u64,
    block_hash: &[u8],
    vote_tracker: &VoteTracker,
    validators: &[ValidatorInfo],
) -> bool {
    let total_stake: u64 = validators.iter().map(|v| v.stake).sum();
    let voting_stake = vote_tracker.get_stake_voting_for(slot, block_hash, validators);
    
    voting_stake * 3 > total_stake * 2
}
```

### Fork Choice

When multiple blocks exist at the same slot, choose the one with most stake:

```rust
pub fn fork_choice(
    slot: u64,
    candidates: &[Block],
    vote_tracker: &VoteTracker,
    validators: &[ValidatorInfo],
) -> Option<Block> {
    let mut best_block = None;
    let mut best_stake = 0u64;
    
    for block in candidates {
        let stake = vote_tracker.get_stake_voting_for(
            slot,
            &block.hash(),
            validators,
        );
        
        if stake > best_stake {
            best_stake = stake;
            best_block = Some(block.clone());
        }
    }
    
    best_block
}
```

## Slashing

### Slashable Offenses

1. **Double Production**: Producing multiple blocks at same slot
2. **Invalid Block**: Including invalid transactions
3. **Censorship**: Consistently excluding valid transactions

### Slash Implementation

```rust
pub fn slash_validator(
    validator: &Pubkey,
    offense: SlashingOffense,
) -> Result<()> {
    let slash_amount = match offense {
        SlashingOffense::DoubleProduction => {
            get_validator_stake(validator)? / 10  // 10% slash
        }
        SlashingOffense::InvalidBlock => {
            get_validator_stake(validator)? / 20  // 5% slash
        }
        SlashingOffense::Censorship => {
            get_validator_stake(validator)? / 100 // 1% slash
        }
    };
    
    reduce_stake(validator, slash_amount)?;
    
    emit!(SlashEvent {
        validator: *validator,
        amount: slash_amount,
        reason: offense,
    });
    
    Ok(())
}
```

## Rewards

### Block Rewards

```rust
pub const BLOCK_REWARD: u64 = 1_000_000; // 0.001 ADE per block

pub fn distribute_rewards(
    slot: u64,
    leader: &Pubkey,
    voters: &[Pubkey],
) -> Result<()> {
    // 80% to leader
    let leader_reward = BLOCK_REWARD * 80 / 100;
    mint_tokens(leader, leader_reward)?;
    
    // 20% to voters
    let voter_reward = BLOCK_REWARD * 20 / 100 / voters.len() as u64;
    for voter in voters {
        mint_tokens(voter, voter_reward)?;
    }
    
    Ok(())
}
```

### Staking Rewards

Annual percentage yield (APY) based on total staked:

```rust
pub fn calculate_staking_apy(total_staked: u64, total_supply: u64) -> f64 {
    let base_apy = 0.08; // 8% base
    let stake_ratio = total_staked as f64 / total_supply as f64;
    
    // APY decreases as stake ratio increases
    base_apy / stake_ratio
}
```

## Network Upgrades

### Epoch Boundaries

Major upgrades occur at epoch boundaries:

```rust
pub fn check_upgrade_activation(epoch: u64) -> Option<Upgrade> {
    match epoch {
        100 => Some(Upgrade::V1_1),
        200 => Some(Upgrade::V1_2),
        _ => None,
    }
}
```

### Feature Gates

```rust
pub struct FeatureGate {
    pub activation_epoch: u64,
    pub activated: bool,
}

pub fn is_feature_active(feature: &str, current_epoch: u64) -> bool {
    let gate = get_feature_gate(feature)?;
    current_epoch >= gate.activation_epoch
}
```

## Performance Metrics

### Theoretical Limits

- **Max TPS**: 50,000
- **Block Time**: 400ms
- **Finality Time**: ~13 seconds (32 blocks)
- **Network Latency Tolerance**: <100ms

### Monitoring

```rust
pub struct ConsensusMetrics {
    pub average_block_time: Duration,
    pub fork_rate: f64,
    pub vote_participation: f64,
    pub validator_count: usize,
}

pub async fn collect_metrics() -> ConsensusMetrics {
    ConsensusMetrics {
        average_block_time: measure_block_time().await,
        fork_rate: calculate_fork_rate().await,
        vote_participation: calculate_participation().await,
        validator_count: count_active_validators().await,
    }
}
```

## Best Practices

### For Validators

1. **Maintain Uptime**: >95% uptime required
2. **Fast Hardware**: NVMe SSD, high-speed network
3. **Monitor Performance**: Track vote success rate
4. **Secure Keys**: Use hardware security modules
5. **Backup**: Maintain hot backup nodes

### For Developers

1. **Wait for Finality**: 32 confirmations for critical operations
2. **Handle Forks**: Implement fork detection
3. **Monitor Validators**: Track validator set changes
4. **Optimize Timing**: Submit transactions early in slot

