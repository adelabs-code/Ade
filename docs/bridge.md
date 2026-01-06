# Bridge Protocol

## Overview

The Ade Bridge enables trustless asset transfers between Solana mainnet and the Ade sidechain. It uses a relayer-based architecture with cryptographic proofs for security.

## Architecture

```
Solana Mainnet              Relayers              Ade Sidechain
      │                        │                        │
      │  Lock Assets           │                        │
      ├───────────────────────►│                        │
      │                        │                        │
      │                        │  Monitor & Relay       │
      │                        ├───────────────────────►│
      │                        │                        │
      │                        │  Mint Wrapped Assets   │
      │                        │◄───────────────────────┤
      │                        │                        │
```

## Components

### Lock Contract (Solana)

Holds assets on Solana mainnet.

```rust
pub struct LockContract {
    pub authority: Pubkey,
    pub locked_funds: HashMap<Pubkey, u64>,
    pub nonce: u64,
}

pub fn lock_assets(
    token: Pubkey,
    amount: u64,
    recipient: Vec<u8>,
) -> Result<DepositEvent> {
    // Verify balance
    // Transfer to lock account
    // Emit deposit event
}
```

### Mint Contract (Ade)

Issues wrapped assets on Ade sidechain.

```rust
pub struct MintContract {
    pub wrapped_tokens: HashMap<Vec<u8>, u64>,
}

pub fn mint_wrapped(
    proof: BridgeProof,
    amount: u64,
    recipient: Vec<u8>,
) -> Result<()> {
    // Verify proof
    // Mint wrapped tokens
    // Update state
}
```

### Relayer

Monitors both chains and relays proofs.

**Configuration:**
```toml
[relayer]
solana_rpc = "https://api.mainnet-beta.solana.com"
ade_rpc = "http://localhost:8899"
poll_interval_ms = 1000
confirmation_threshold = 32
```

**Operation:**
```rust
async fn relay_loop() {
    loop {
        // Poll Solana for deposits
        let deposits = poll_solana_deposits().await?;
        
        for deposit in deposits {
            // Generate proof
            let proof = generate_proof(&deposit)?;
            
            // Submit to Ade
            submit_to_ade(proof).await?;
        }
        
        // Poll Ade for withdrawals
        let withdrawals = poll_ade_withdrawals().await?;
        
        for withdrawal in withdrawals {
            let proof = generate_proof(&withdrawal)?;
            submit_to_solana(proof).await?;
        }
        
        sleep(Duration::from_millis(1000)).await;
    }
}
```

## Deposit Flow

### 1. User Initiates Deposit

```typescript
import { BridgeClient } from 'ade-sidechain';

const bridge = new BridgeClient(client);

const result = await bridge.deposit(
  'solana',              // From chain
  1_000_000_000,         // Amount (1 SOL)
  'So11111111...'        // Token address
);

console.log(`Deposit ID: ${result.depositId}`);
```

### 2. Lock Assets on Solana

```rust
// Solana program instruction
pub fn process_lock(
    accounts: &[AccountInfo],
    amount: u64,
    recipient: Vec<u8>,
) -> ProgramResult {
    let source = next_account_info(accounts_iter)?;
    let lock_account = next_account_info(accounts_iter)?;
    
    // Transfer tokens
    token::transfer(
        source,
        lock_account,
        amount,
    )?;
    
    // Emit event
    emit!(DepositEvent {
        depositor: *source.key,
        amount,
        recipient,
        nonce: get_and_increment_nonce(),
    });
    
    Ok(())
}
```

### 3. Relayer Detects Event

```rust
async fn poll_solana_deposits() -> Result<Vec<DepositEvent>> {
    let signatures = client
        .get_signatures_for_address(&lock_contract_address)
        .await?;
    
    let mut deposits = Vec::new();
    
    for sig in signatures {
        let tx = client.get_transaction(&sig.signature).await?;
        
        if let Some(event) = parse_deposit_event(&tx) {
            deposits.push(event);
        }
    }
    
    Ok(deposits)
}
```

### 4. Submit Proof to Ade

```rust
pub struct BridgeProof {
    pub source_chain: String,
    pub tx_hash: Vec<u8>,
    pub block_number: u64,
    pub merkle_proof: Vec<Vec<u8>>,
    pub event_data: Vec<u8>,
    pub signatures: Vec<Vec<u8>>,
}

async fn submit_to_ade(proof: BridgeProof) -> Result<String> {
    let client = AdeClient::new(ade_rpc_url);
    
    let signature = client
        .submit_bridge_proof(proof)
        .await?;
    
    Ok(signature)
}
```

### 5. Mint Wrapped Assets

```rust
pub fn process_bridge_deposit(
    proof: BridgeProof,
) -> Result<()> {
    // Verify proof
    verify_bridge_proof(&proof)?;
    
    // Parse event
    let event: DepositEvent = bincode::deserialize(&proof.event_data)?;
    
    // Mint wrapped tokens
    mint_wrapped_tokens(
        &event.recipient,
        event.amount,
    )?;
    
    Ok(())
}
```

## Withdrawal Flow

### 1. User Initiates Withdrawal

```python
from ade_sidechain import BridgeClient

bridge = BridgeClient(client)

result = bridge.withdraw(
    to_chain='solana',
    amount=500_000_000,  # 0.5 SOL
    recipient='YourSolanaAddress...'
)
```

### 2. Burn Wrapped Assets

```rust
pub fn process_withdrawal(
    from: &Pubkey,
    amount: u64,
    recipient: Vec<u8>,
) -> Result<()> {
    // Verify balance
    let account = get_account(from)?;
    require!(account.balance >= amount, InsufficientBalance);
    
    // Burn tokens
    burn_wrapped_tokens(from, amount)?;
    
    // Emit event
    emit!(WithdrawalEvent {
        withdrawer: *from,
        amount,
        recipient,
        nonce: get_and_increment_nonce(),
    });
    
    Ok(())
}
```

### 3. Relayer Submits to Solana

```rust
async fn submit_to_solana(proof: BridgeProof) -> Result<Signature> {
    let client = SolanaClient::new(solana_rpc_url);
    
    let ix = unlock_instruction(proof)?;
    
    let tx = Transaction::new_signed_with_payer(
        &[ix],
        Some(&relayer_keypair.pubkey()),
        &[&relayer_keypair],
        recent_blockhash,
    );
    
    let signature = client.send_transaction(&tx).await?;
    
    Ok(signature)
}
```

### 4. Unlock Assets on Solana

```rust
pub fn process_unlock(
    accounts: &[AccountInfo],
    proof: BridgeProof,
) -> ProgramResult {
    // Verify proof
    verify_proof(&proof)?;
    
    let event: WithdrawalEvent = parse_event(&proof)?;
    
    // Transfer from lock account to recipient
    token::transfer(
        lock_account,
        recipient_account,
        event.amount,
    )?;
    
    Ok(())
}
```

## Security

### Proof Verification

```rust
pub fn verify_bridge_proof(proof: &BridgeProof) -> Result<()> {
    // 1. Verify block finality
    require!(
        proof.block_number + FINALITY_THRESHOLD <= current_block(),
        BlockNotFinalized
    );
    
    // 2. Verify merkle proof
    let root = compute_merkle_root(&proof.merkle_proof)?;
    let expected_root = get_block_root(proof.block_number)?;
    require!(root == expected_root, InvalidMerkleProof);
    
    // 3. Verify relayer signatures
    verify_relayer_signatures(&proof.signatures, &proof.event_data)?;
    
    // 4. Check for replay
    require!(!is_proof_used(&proof.tx_hash), ProofAlreadyUsed);
    
    Ok(())
}
```

### Multi-Sig Relayer

```rust
pub struct RelayerSet {
    pub relayers: Vec<Pubkey>,
    pub threshold: usize,
}

pub fn verify_relayer_signatures(
    signatures: &[Vec<u8>],
    message: &[u8],
) -> Result<()> {
    let relayer_set = get_relayer_set()?;
    
    require!(
        signatures.len() >= relayer_set.threshold,
        InsufficientSignatures
    );
    
    let mut valid_count = 0;
    
    for (sig, relayer) in signatures.iter().zip(&relayer_set.relayers) {
        if verify_signature(sig, message, relayer)? {
            valid_count += 1;
        }
    }
    
    require!(
        valid_count >= relayer_set.threshold,
        ThresholdNotMet
    );
    
    Ok(())
}
```

### Fraud Proofs

```rust
pub fn submit_fraud_proof(
    challenged_proof: BridgeProof,
    fraud_evidence: Vec<u8>,
) -> Result<()> {
    // Verify fraud evidence
    let is_fraud = verify_fraud_evidence(&challenged_proof, &fraud_evidence)?;
    
    if is_fraud {
        // Slash relayers
        slash_relayers(&challenged_proof.signatures)?;
        
        // Revert transaction
        revert_bridge_transaction(&challenged_proof)?;
    }
    
    Ok(())
}
```

## Usage Examples

### TypeScript

```typescript
import { AdeSidechainClient, BridgeClient } from 'ade-sidechain';

const client = new AdeSidechainClient('http://localhost:8899');
const bridge = new BridgeClient(client);

// Deposit
const deposit = await bridge.deposit(
  'solana',
  1_000_000_000,
  'So11111111111111111111111111111111111111112'
);

console.log(`Deposit ID: ${deposit.depositId}`);

// Check status
const status = await bridge.getDepositStatus(deposit.depositId);
console.log(`Status: ${status}`);

// Withdraw
const withdrawal = await bridge.withdraw(
  'solana',
  500_000_000,
  'YourAddress...'
);
```

### Python

```python
from ade_sidechain import AdeSidechainClient, BridgeClient

client = AdeSidechainClient('http://localhost:8899')
bridge = BridgeClient(client)

# Deposit from Solana
deposit = bridge.deposit(
    from_chain='solana',
    amount=1_000_000_000,
    token_address='So11111111111111111111111111111111111111112'
)

print(f"Deposit ID: {deposit['depositId']}")

# Withdraw to Solana
withdrawal = bridge.withdraw(
    to_chain='solana',
    amount=500_000_000,
    recipient='YourSolanaAddress...'
)

print(f"Withdrawal ID: {withdrawal['withdrawalId']}")
```

## Monitoring

### Bridge Events

```rust
pub enum BridgeEvent {
    Deposit {
        id: Vec<u8>,
        from_chain: String,
        amount: u64,
        token: Vec<u8>,
    },
    Withdrawal {
        id: Vec<u8>,
        to_chain: String,
        amount: u64,
        recipient: Vec<u8>,
    },
    ProofSubmitted {
        proof_hash: Vec<u8>,
        relayer: Pubkey,
    },
    FraudDetected {
        proof_hash: Vec<u8>,
        evidence: Vec<u8>,
    },
}
```

### Metrics

```bash
curl http://localhost:8899/bridge/metrics
```

Response:
```json
{
  "total_deposits": 1234,
  "total_withdrawals": 567,
  "total_volume": 98765432100,
  "active_relayers": 5,
  "pending_deposits": 3,
  "pending_withdrawals": 1
}
```

## Best Practices

1. **Wait for Finality**: Ensure sufficient confirmations before considering deposit complete
2. **Monitor Events**: Subscribe to bridge events for real-time updates
3. **Error Handling**: Implement retry logic for failed relays
4. **Gas Management**: Keep sufficient balance for transaction fees
5. **Security**: Use multi-sig for high-value transfers
6. **Testing**: Test on devnet before mainnet usage








