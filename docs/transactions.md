# Transaction Format and Specification

## Overview

Ade transactions follow a Solana-inspired format with extensions for AI-specific operations.

## Transaction Structure

```rust
pub struct Transaction {
    pub signatures: Vec<Vec<u8>>,  // Ed25519 signatures
    pub message: Message,
}

pub struct Message {
    pub header: MessageHeader,
    pub account_keys: Vec<Vec<u8>>,
    pub recent_blockhash: Vec<u8>,
    pub instructions: Vec<Instruction>,
}

pub struct MessageHeader {
    pub num_required_signatures: u8,
    pub num_readonly_signed_accounts: u8,
    pub num_readonly_unsigned_accounts: u8,
}
```

## Instruction Types

### Standard Instructions

#### Transfer

Transfer tokens between accounts.

```rust
pub struct TransferInstruction {
    pub from: Vec<u8>,
    pub to: Vec<u8>,
    pub amount: u64,
}
```

**Example:**
```typescript
const instruction = TransferInstruction.create(
  fromPubkey,
  toPubkey,
  1_000_000_000n,  // 1 token
  programId
);
```

#### CreateAccount

Create a new account.

```rust
pub struct CreateAccountInstruction {
    pub owner: Vec<u8>,
    pub space: u64,
    pub lamports: u64,
}
```

### AI Agent Instructions

#### AIAgentDeploy

Deploy a new AI agent.

```rust
pub struct AIAgentDeployInstruction {
    pub agent_id: Vec<u8>,
    pub model_hash: Vec<u8>,
    pub config: Vec<u8>,
}
```

**Binary Format:**
```
[1 byte]  Instruction discriminator (0x02)
[32 bytes] Agent ID
[1 byte]   Model hash length
[N bytes]  Model hash
[2 bytes]  Config length
[M bytes]  Config JSON
```

**Example:**
```python
from ade_sidechain.transaction import TransactionBuilder, CompiledInstruction

instruction = CompiledInstruction(
    program_id_index=0,
    accounts=[0, 1],
    data=encode_ai_agent_deploy(
        agent_id=b'my-agent-001',
        model_hash=b'QmXxYyZz...',
        config={'modelType': 'transformer'}
    )
)
```

#### AIAgentExecute

Execute an AI agent.

```rust
pub struct AIAgentExecuteInstruction {
    pub agent_id: Vec<u8>,
    pub input_data: Vec<u8>,
    pub max_compute: u64,
}
```

**Binary Format:**
```
[1 byte]   Instruction discriminator (0x03)
[32 bytes] Agent ID
[2 bytes]  Input data length
[N bytes]  Input data
[8 bytes]  Max compute units
```

### Bridge Instructions

#### BridgeDeposit

Initiate a bridge deposit.

```rust
pub struct BridgeDepositInstruction {
    pub from_chain: String,
    pub amount: u64,
    pub token_address: Vec<u8>,
}
```

**Binary Format:**
```
[1 byte]   Instruction discriminator (0x05)
[1 byte]   Chain name length
[N bytes]  Chain name
[8 bytes]  Amount
[32 bytes] Token address
```

#### BridgeWithdraw

Initiate a bridge withdrawal.

```rust
pub struct BridgeWithdrawInstruction {
    pub to_chain: String,
    pub amount: u64,
    pub recipient: Vec<u8>,
}
```

## Transaction Building

### TypeScript

```typescript
import { TransactionBuilder, Transaction } from 'ade-sidechain';

const builder = new TransactionBuilder();

const transaction = builder
  .addInstruction(instruction1)
  .addInstruction(instruction2)
  .setRecentBlockhash(blockhash)
  .build();

transaction.sign([keypair]);

const serialized = transaction.serialize();
```

### Python

```python
from ade_sidechain.transaction import TransactionBuilder, CompiledInstruction

builder = TransactionBuilder()

tx = (builder
    .add_instruction(instruction1)
    .add_instruction(instruction2)
    .set_recent_blockhash(blockhash)
    .build())

serialized = tx.serialize()
```

### Rust

```rust
use ade_transaction::{Transaction, TransactionBuilder};

let tx = TransactionBuilder::new()
    .add_instruction(instruction)
    .add_signer(keypair)
    .set_recent_blockhash(blockhash)
    .build()?;

let serialized = tx.serialize()?;
```

## Validation

### Signature Verification

```rust
pub fn verify_signatures(tx: &Transaction) -> Result<bool> {
    let message_bytes = bincode::serialize(&tx.message)?;
    
    for (i, sig_bytes) in tx.signatures.iter().enumerate() {
        let pubkey = &tx.message.account_keys[i];
        let signature = Signature::from_bytes(sig_bytes)?;
        
        PublicKey::from_bytes(pubkey)?
            .verify_strict(&message_bytes, &signature)?;
    }
    
    Ok(true)
}
```

### Balance Verification

```rust
pub fn verify_balance(
    account: &Account,
    required: u64,
) -> Result<()> {
    if account.lamports < required {
        return Err(Error::InsufficientBalance {
            required,
            available: account.lamports,
        });
    }
    Ok(())
}
```

### Compute Budget Verification

```rust
pub fn verify_compute_budget(
    instructions: &[Instruction],
    max_compute: u64,
) -> Result<()> {
    let total_compute: u64 = instructions
        .iter()
        .map(|ix| estimate_compute_cost(ix))
        .sum();
    
    if total_compute > max_compute {
        return Err(Error::ComputeBudgetExceeded(total_compute));
    }
    
    Ok(())
}
```

## Serialization

### Wire Format

```
Transaction:
[1 byte]   Number of signatures
[64 bytes] * num_signatures
[Message bytes]

Message:
[3 bytes]  Header
[1+ bytes] Number of account keys (compact-u16)
[32 bytes] * num_account_keys
[32 bytes] Recent blockhash
[1+ bytes] Number of instructions (compact-u16)
[Instruction bytes] * num_instructions

Instruction:
[1 byte]   Program ID index
[1+ bytes] Number of accounts (compact-u16)
[1 byte]   * num_accounts (account indices)
[1+ bytes] Data length (compact-u16)
[N bytes]  Data
```

### Compact-u16 Encoding

```rust
fn encode_compact_u16(len: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut len = len;
    
    while len > 0x7f {
        bytes.push((len & 0x7f) | 0x80);
        len >>= 7;
    }
    bytes.push(len as u8);
    
    bytes
}
```

## Transaction Lifecycle

```
1. Build      → Construct transaction object
2. Sign       → Add signatures
3. Serialize  → Convert to bytes
4. Submit     → Send to RPC
5. Validate   → Node validates
6. Execute    → Process instructions
7. Confirm    → Block inclusion
8. Finalize   → 32 confirmations
```

## Best Practices

### 1. Recent Blockhash

Always use a fresh blockhash (< 150 blocks old):

```typescript
const { blockhash } = await client.getLatestBlockhash();
builder.setRecentBlockhash(Buffer.from(blockhash));
```

### 2. Fee Estimation

```typescript
const estimatedFee = transaction.estimateFee();
console.log(`Estimated fee: ${estimatedFee} lamports`);
```

### 3. Retry Logic

```python
def send_with_retry(client, tx, max_retries=3):
    for attempt in range(max_retries):
        try:
            signature = client.send_transaction(tx)
            return signature
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
```

### 4. Transaction Size

Keep transactions under 1232 bytes:

```rust
const MAX_TX_SIZE: usize = 1232;

if tx.serialize()?.len() > MAX_TX_SIZE {
    return Err(Error::TransactionTooLarge);
}
```

### 5. Instruction Ordering

Order instructions carefully for dependencies:

```typescript
builder
  .addInstruction(createAccountIx)   // Must come first
  .addInstruction(initializeIx)      // Depends on account
  .addInstruction(executeIx);        // Depends on initialization
```

## Error Handling

```rust
pub enum TransactionError {
    InvalidSignature,
    InsufficientBalance { required: u64, available: u64 },
    InvalidAccount(String),
    ComputeBudgetExceeded(u64),
    TransactionTooLarge,
    BlockhashExpired,
}
```

**Example:**
```python
try:
    signature = client.send_transaction(tx)
except RpcError as e:
    if e.code == -32002:  # Insufficient balance
        print("Insufficient balance")
    elif e.code == -32003:  # Invalid signature
        print("Invalid signature")
    else:
        print(f"Unknown error: {e.message}")
```

## Advanced Topics

### Priority Fees

```rust
pub struct PriorityFeeInstruction {
    pub microlamports_per_compute_unit: u64,
}
```

### Versioned Transactions

Support for future transaction versions:

```rust
pub enum VersionedTransaction {
    V0(TransactionV0),
    V1(TransactionV1),  // Future
}
```

### Atomic Transactions

All instructions succeed or all fail:

```rust
pub fn execute_transaction(tx: &Transaction) -> Result<()> {
    let mut state_backup = backup_state();
    
    for instruction in &tx.message.instructions {
        if let Err(e) = execute_instruction(instruction) {
            restore_state(state_backup);
            return Err(e);
        }
    }
    
    commit_state();
    Ok(())
}
```


