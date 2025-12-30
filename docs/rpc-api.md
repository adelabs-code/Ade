# RPC API Reference

## Overview

The Ade RPC API follows JSON-RPC 2.0 specification and provides methods for interacting with the sidechain.

**Endpoint:** `http://localhost:8899` (default)

## Request Format

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "methodName",
  "params": { }
}
```

## Response Format

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": { }
}
```

## Core Methods

### getSlot

Returns the current slot number.

**Parameters:** None

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": 12345678
}
```

**Example:**
```bash
curl -X POST http://localhost:8899 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"getSlot"}'
```

### getBlockHeight

Returns the current block height.

**Parameters:** None

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": 12345678
}
```

### getLatestBlockhash

Returns the latest blockhash.

**Parameters:** None

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "blockhash": "4sGjMW1sUnHzSxGspuhpqLDx6wiyjNtZ",
    "lastValidBlockHeight": 12345828
  }
}
```

### sendTransaction

Submits a signed transaction.

**Parameters:**
- `transaction` (string): Base58-encoded signed transaction

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": "2id3YC2jK9G5Wo2phDx4gJVAew8DcY5NAojnVuao8rkxwPYPe8cSwE5GzhEgJA2y8fVjDEo6iR6ykBvDxrTQrtpb"
}
```

**Example:**
```typescript
const signature = await client.sendTransaction(
  encodedTransaction
);
```

### getTransaction

Retrieves transaction details by signature.

**Parameters:**
- `signature` (string): Transaction signature

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "slot": 100,
    "transaction": { },
    "meta": {
      "err": null,
      "status": { "Ok": null }
    }
  }
}
```

### getBalance

Returns the balance for a given address.

**Parameters:**
- `address` (string): Account address

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "value": 1000000000
  }
}
```

### getAccountInfo

Returns account information.

**Parameters:**
- `address` (string): Account address

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "lamports": 1000000000,
    "owner": "11111111111111111111111111111111",
    "executable": false,
    "rentEpoch": 0,
    "data": ""
  }
}
```

## AI Agent Methods

### deployAIAgent

Deploys a new AI agent to the sidechain.

**Parameters:**
```json
{
  "agentId": "my-agent-001",
  "modelHash": "QmXxYyZz1234567890abcdef",
  "config": {
    "modelType": "transformer",
    "parameters": {
      "maxTokens": 512,
      "temperature": 0.7
    },
    "maxExecutionTime": 30000,
    "allowedOperations": ["inference"]
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "agentId": "my-agent-001",
    "signature": "3nkSQ9hYdCXdJd7mY7XMhXnk..."
  }
}
```

**Example:**
```python
result = client.deploy_ai_agent(
    agent_id='my-agent-001',
    model_hash='QmXxYyZz...',
    config=config
)
```

### executeAIAgent

Executes an AI agent with input data.

**Parameters:**
```json
{
  "agentId": "my-agent-001",
  "inputData": {
    "prompt": "What is the meaning of life?",
    "maxLength": 100
  },
  "maxCompute": 50000
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "executionId": "exec_12345",
    "signature": "2id3YC2jK9G5Wo2phDx4g...",
    "computeUnits": 45000,
    "output": { }
  }
}
```

### getAIAgentInfo

Retrieves information about an AI agent.

**Parameters:**
```json
{
  "agentId": "my-agent-001"
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "agentId": "my-agent-001",
    "modelHash": "QmXxYyZz...",
    "owner": "ownerpubkey...",
    "executionCount": 42,
    "totalComputeUsed": 2100000
  }
}
```

## Bridge Methods

### bridgeDeposit

Initiates a deposit from another chain.

**Parameters:**
```json
{
  "fromChain": "solana",
  "amount": 1000000000,
  "tokenAddress": "So11111111111111111111111111111111111111112"
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "depositId": "deposit_12345",
    "signature": "3nkSQ9hYdCXdJd7mY7XMhXnk..."
  }
}
```

### bridgeWithdraw

Initiates a withdrawal to another chain.

**Parameters:**
```json
{
  "toChain": "solana",
  "amount": 500000000,
  "recipient": "YourSolanaAddressHere..."
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "withdrawalId": "withdraw_12345",
    "signature": "2id3YC2jK9G5Wo2phDx4g..."
  }
}
```

## Utility Methods

### health

Health check endpoint.

**Parameters:** None

**Response:** "OK"

**Example:**
```bash
curl http://localhost:8899/health
```

### metrics

Returns node metrics.

**Parameters:** None

**Response:**
```json
{
  "slot": 12345678,
  "transaction_count": 9876543
}
```

## Error Codes

| Code | Message | Description |
|------|---------|-------------|
| -32600 | Invalid Request | Invalid JSON-RPC request |
| -32601 | Method not found | Method does not exist |
| -32602 | Invalid params | Invalid method parameters |
| -32603 | Internal error | Internal JSON-RPC error |
| -32700 | Parse error | Invalid JSON |

**Error Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32600,
    "message": "Invalid Request"
  }
}
```

## Rate Limiting

- **Default Limit**: 100 requests per second per IP
- **Burst Limit**: 200 requests
- **Headers**: `X-RateLimit-Remaining`, `X-RateLimit-Reset`

## WebSocket Support

WebSocket endpoint for real-time updates (planned):

```javascript
const ws = new WebSocket('ws://localhost:8900');

ws.send(JSON.stringify({
  method: 'subscribe',
  params: ['slot']
}));
```

## SDK Examples

### TypeScript
```typescript
import { AdeSidechainClient } from 'ade-sidechain';

const client = new AdeSidechainClient('http://localhost:8899');
const slot = await client.getSlot();
```

### Python
```python
from ade_sidechain import AdeSidechainClient

client = AdeSidechainClient('http://localhost:8899')
slot = client.get_slot()
```

### Rust
```rust
use ade_rpc::RpcClient;

let client = RpcClient::new("http://localhost:8899");
let slot = client.get_slot().await?;
```

