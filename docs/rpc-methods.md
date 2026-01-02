# Complete RPC Methods Reference

## Overview

Ade Sidechain provides 50+ RPC methods for comprehensive blockchain interaction.

## Method Categories

1. **Block Methods** (9 methods) - Block and slot information
2. **Transaction Methods** (7 methods) - Transaction submission and queries
3. **Account Methods** (7 methods) - Account and token operations
4. **Validator & Staking Methods** (7 methods) - Validator and staking info
5. **Network & Cluster Methods** (8 methods) - Network and cluster data
6. **AI Agent Methods** (6 methods) - AI agent deployment and execution
7. **Bridge Methods** (5 methods) - Cross-chain bridge operations
8. **Utility Methods** (7 methods) - Helper and utility functions

## Block Methods

### getSlot
Get the current slot.

**Parameters:** None

**Response:** `number`

```bash
curl -X POST http://localhost:8899 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"getSlot"}'
```

### getBlockHeight
Get current block height.

**Parameters:** None

**Response:** `number`

### getBlock
Get block information for a specific slot.

**Parameters:**
- `slot` (number): Block slot number

**Response:** Block object with transactions and metadata

### getBlocks
Get a list of confirmed blocks between two slots.

**Parameters:**
- `startSlot` (number): Start slot
- `endSlot` (number, optional): End slot

**Response:** Array of slot numbers

### getBlockTime
Get estimated production time of a block.

**Parameters:**
- `slot` (number): Block slot

**Response:** Unix timestamp

### getFirstAvailableBlock
Get the first available block.

**Parameters:** None

**Response:** Slot number

### getLatestBlockhash
Get latest blockhash.

**Parameters:** None

**Response:**
```json
{
  "blockhash": "string",
  "lastValidBlockHeight": number
}
```

### getBlockProduction
Get block production information.

**Parameters:** None

**Response:** Block production stats by validator

### getBlockCommitment
Get commitment for a particular block.

**Parameters:**
- `slot` (number): Block slot

**Response:** Commitment information

## Transaction Methods

### sendTransaction
Submit a signed transaction.

**Parameters:**
- `transaction` (string): Signed transaction (base64)

**Response:** Transaction signature

```typescript
const signature = await client.sendTransaction(encodedTx);
```

### simulateTransaction
Simulate a transaction without committing.

**Parameters:**
- `transaction` (string): Transaction to simulate

**Response:**
```json
{
  "err": null,
  "logs": ["..."],
  "unitsConsumed": number,
  "accounts": null
}
```

### getTransaction
Get transaction details by signature.

**Parameters:**
- `signature` (string): Transaction signature

**Response:** Full transaction object with metadata

### getTransactionCount
Get total transaction count.

**Parameters:** None

**Response:** `number`

### getRecentPerformanceSamples
Get recent performance samples.

**Parameters:**
- `limit` (number, optional): Number of samples (default: 720)

**Response:** Array of performance samples

### getSignatureStatuses
Get statuses of a list of signatures.

**Parameters:**
- `signatures` (string[]): Array of signatures

**Response:** Array of signature statuses

### getSignaturesForAddress
Get confirmed signatures for an address.

**Parameters:**
- `address` (string): Account address
- `limit` (number, optional): Max signatures to return

**Response:** Array of signature info

## Account Methods

### getBalance
Get account balance.

**Parameters:**
- `address` (string): Account public key

**Response:**
```json
{
  "context": { "slot": number },
  "value": number
}
```

### getAccountInfo
Get account information.

**Parameters:**
- `address` (string): Account public key

**Response:** Account object

### getMultipleAccounts
Get information about multiple accounts.

**Parameters:**
- `addresses` (string[]): Array of public keys

**Response:** Array of account objects

### getProgramAccounts
Get all accounts owned by a program.

**Parameters:**
- `programId` (string): Program public key

**Response:** Array of accounts

### getLargestAccounts
Get largest accounts by balance.

**Parameters:** None

**Response:** Array of top accounts

### getTokenAccountsByOwner
Get token accounts by owner.

**Parameters:**
- `owner` (string): Owner public key
- `mint` (string, optional): Token mint

**Response:** Array of token accounts

### getTokenSupply
Get token supply information.

**Parameters:**
- `mint` (string): Token mint address

**Response:** Token supply details

## Validator & Staking Methods

### getVoteAccounts
Get vote accounts.

**Parameters:** None

**Response:**
```json
{
  "current": [...],
  "delinquent": [...]
}
```

### getValidators
Get current validators.

**Parameters:** None

**Response:** Validator list

### getStakeActivation
Get stake activation state.

**Parameters:**
- `stakeAccount` (string): Stake account pubkey

**Response:**
```json
{
  "state": "active" | "inactive" | "activating" | "deactivating",
  "active": number,
  "inactive": number
}
```

### getStakeMinimumDelegation
Get minimum stake delegation amount.

**Parameters:** None

**Response:** `number`

### getLeaderSchedule
Get leader schedule for an epoch.

**Parameters:**
- `slot` (number, optional): Slot to query

**Response:** Map of validator to slot numbers

### getEpochInfo
Get current epoch information.

**Parameters:** None

**Response:**
```json
{
  "absoluteSlot": number,
  "blockHeight": number,
  "epoch": number,
  "slotIndex": number,
  "slotsInEpoch": number,
  "transactionCount": number
}
```

### getEpochSchedule
Get epoch schedule.

**Parameters:** None

**Response:** Epoch schedule configuration

## Network & Cluster Methods

### getClusterNodes
Get cluster nodes.

**Parameters:** None

**Response:** Array of node information

### getVersion
Get node version.

**Parameters:** None

**Response:**
```json
{
  "solana-core": "string",
  "feature-set": number
}
```

### getGenesisHash
Get genesis hash.

**Parameters:** None

**Response:** Genesis hash string

### getIdentity
Get node identity.

**Parameters:** None

**Response:** Node identity pubkey

### getInflationGovernor
Get inflation governor.

**Parameters:** None

**Response:** Inflation parameters

### getInflationRate
Get current inflation rate.

**Parameters:** None

**Response:** Inflation rates

### getInflationReward
Get inflation rewards for addresses.

**Parameters:**
- `addresses` (string[]): Array of addresses
- `epoch` (number, optional): Epoch number

**Response:** Array of rewards

### getSupply
Get token supply information.

**Parameters:** None

**Response:**
```json
{
  "context": { "slot": number },
  "value": {
    "total": number,
    "circulating": number,
    "nonCirculating": number,
    "nonCirculatingAccounts": []
  }
}
```

## AI Agent Methods

### deployAIAgent
Deploy a new AI agent.

**Parameters:**
```json
{
  "agentId": "string",
  "modelHash": "string",
  "config": {
    "modelType": "string",
    "parameters": {},
    "maxExecutionTime": number,
    "allowedOperations": []
  }
}
```

**Response:**
```json
{
  "agentId": "string",
  "signature": "string"
}
```

### executeAIAgent
Execute an AI agent.

**Parameters:**
```json
{
  "agentId": "string",
  "inputData": any,
  "maxCompute": number
}
```

**Response:**
```json
{
  "executionId": "string",
  "signature": "string",
  "computeUnits": number,
  "output": {}
}
```

### getAIAgentInfo
Get AI agent information.

**Parameters:**
- `agentId` (string): Agent ID

**Response:** Agent metadata and stats

### updateAIAgent
Update AI agent configuration.

**Parameters:**
- `agentId` (string): Agent ID
- `newConfig` (object): New configuration

**Response:** Transaction signature

### listAIAgents
List AI agents.

**Parameters:**
- `owner` (string, optional): Filter by owner

**Response:** Array of agents

### getAIAgentExecutions
Get execution history for an agent.

**Parameters:**
- `agentId` (string): Agent ID

**Response:** Array of executions

## Bridge Methods

### bridgeDeposit
Initiate cross-chain deposit.

**Parameters:**
```json
{
  "fromChain": "string",
  "amount": number,
  "tokenAddress": "string"
}
```

**Response:**
```json
{
  "depositId": "string",
  "signature": "string",
  "status": "pending"
}
```

### bridgeWithdraw
Initiate cross-chain withdrawal.

**Parameters:**
```json
{
  "toChain": "string",
  "amount": number,
  "recipient": "string"
}
```

**Response:**
```json
{
  "withdrawalId": "string",
  "signature": "string",
  "status": "pending"
}
```

### getBridgeStatus
Get bridge operation status.

**Parameters:**
- `id` (string): Operation ID

**Response:** Operation status and details

### getBridgeHistory
Get bridge transaction history.

**Parameters:**
- `address` (string, optional): Filter by address

**Response:** Array of operations

### estimateBridgeFee
Estimate bridge fees.

**Parameters:**
```json
{
  "fromChain": "string",
  "toChain": "string",
  "amount": number
}
```

**Response:**
```json
{
  "baseFee": number,
  "percentageFee": number,
  "totalFee": number,
  "estimatedTime": number
}
```

## Utility Methods

### requestAirdrop
Request an airdrop of tokens.

**Parameters:**
- `address` (string): Recipient address
- `lamports` (number): Amount to airdrop

**Response:** Transaction signature

### minimumLedgerSlot
Get minimum ledger slot.

**Parameters:** None

**Response:** Slot number

### getSlotLeaders
Get slot leaders.

**Parameters:**
- `start` (number): Start slot
- `limit` (number): Number of leaders

**Response:** Array of validator pubkeys

### getFeeForMessage
Get fee for a message.

**Parameters:**
- `message` (string): Message to estimate

**Response:**
```json
{
  "context": { "slot": number },
  "value": number
}
```

### getRecentPrioritizationFees
Get recent prioritization fees.

**Parameters:** None

**Response:** Array of fee samples

### getMaxRetransmitSlot
Get max retransmit slot.

**Parameters:** None

**Response:** Slot number

### getMaxShredInsertSlot
Get max shred insert slot.

**Parameters:** None

**Response:** Slot number

## Error Codes

| Code | Message | Description |
|------|---------|-------------|
| -32600 | Invalid Request | Invalid JSON-RPC |
| -32601 | Method not found | Unknown method |
| -32602 | Invalid params | Bad parameters |
| -32603 | Internal error | Server error |

## Rate Limits

- **Default**: 100 req/sec per IP
- **Burst**: 200 requests
- **Headers**: `X-RateLimit-Remaining`, `X-RateLimit-Reset`

## SDK Usage

### TypeScript
```typescript
import { createAdeClient, RpcMethod } from 'ade-sidechain';

const client = createAdeClient({ rpcUrl: 'http://localhost:8899' });

// Type-safe calls
const slot = await client.getSlot();
const balance = await client.getBalance('address');
const tx = await client.sendTransaction(encodedTx);
```

### Python
```python
from ade_sidechain import AdeSidechainClient

client = AdeSidechainClient('http://localhost:8899')

slot = client.get_slot()
balance = client.get_balance('address')
```


