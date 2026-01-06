# Ade Sidechain

<div align="center">
  <img src="docs/maina.png" alt="Ade Sidechain" width="400"/>
  
  [![Twitter Follow](https://img.shields.io/twitter/follow/Ade__labs?style=social)](https://twitter.com/Ade__labs)
</div>

A Solana-based sidechain optimized for AI agent deployment and execution.

## Overview

Ade is a high-performance sidechain that extends Solana's capabilities to provide specialized infrastructure for AI agents. It features:

- **AI Agent Runtime**: Deploy and execute AI models (ONNX, PyTorch) on-chain with compute metering
- **Cross-chain Bridge**: Seamless asset transfer between Solana and Ade with Merkle proof verification
- **Proof-of-Stake Consensus**: VRF-based leader selection with VDF-enhanced randomness
- **High Throughput**: 10,000+ TPS with 400ms block time
- **Multi-language SDKs**: TypeScript, Python, Rust, and C/C++ support
- **Production-Ready**: RocksDB storage, LRU caching, dynamic fee markets

## Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                          Ade Sidechain                                 │
├───────────────────────────────────────────────────────────────────────┤
│                           Client Layer                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │ TypeScript  │  │   Python    │  │    Rust     │  │    CLI      │  │
│  │    SDK      │  │    SDK      │  │    SDK      │  │   Tool      │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │
├───────────────────────────────────────────────────────────────────────┤
│                           RPC Layer                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                    │
│  │  JSON-RPC   │  │  WebSocket  │  │   State     │                    │
│  │   Server    │  │   Server    │  │   Backend   │                    │
│  └─────────────┘  └─────────────┘  └─────────────┘                    │
├───────────────────────────────────────────────────────────────────────┤
│                           Node Layer                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │ Validator│ │  Block   │ │ Mempool  │ │  State   │ │   Fee    │    │
│  │          │ │ Producer │ │          │ │Transition│ │  Market  │    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │    AI    │ │   ONNX   │ │ PyTorch  │ │ Compute  │ │ Snapshot │    │
│  │ Runtime  │ │Inference │ │Inference │ │  Meter   │ │ Manager  │    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘    │
├───────────────────────────────────────────────────────────────────────┤
│                         Consensus Layer                                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐    │
│  │  Proof-of-Stake  │  │   VRF Leader     │  │  Finality Gadget │    │
│  │  (Wesolowski VDF)│  │    Selection     │  │                  │    │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘    │
├───────────────────────────────────────────────────────────────────────┤
│                          Bridge Layer                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │ Relayer  │ │  Merkle  │ │  Proof   │ │ MultiSig │ │  Event   │    │
│  │          │ │  Proofs  │ │Verifier  │ │          │ │  Parser  │    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘    │
├───────────────────────────────────────────────────────────────────────┤
│                         Storage Layer                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐    │
│  │     RocksDB      │  │    LRU Cache     │  │    Indexer       │    │
│  │  (Persistent)    │  │   (Hot Data)     │  │  (Secondary)     │    │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘    │
├───────────────────────────────────────────────────────────────────────┤
│                         Network Layer                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐    │
│  │   Gossipsub      │  │ Peer Discovery   │  │  TCP Transport   │    │
│  │   Protocol       │  │   (DHT-like)     │  │                  │    │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘    │
└───────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│                        Solana Mainnet                                  │
│  ┌──────────────────────────┐  ┌──────────────────────────┐          │
│  │   solana-lock Program    │  │   ade-mint Program       │          │
│  │   (Token Locking)        │  │   (Wrapped Token Mint)   │          │
│  └──────────────────────────┘  └──────────────────────────┘          │
└───────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Rust 1.70+
- Node.js 18+
- Python 3.9+ (for Python SDK)

### Build from Source

```bash
# Clone repository
git clone https://github.com/adelabs-code/Ade.git
cd ade

# Build all Rust components
cargo build --release

# Install TypeScript dependencies
npm install

# Build TypeScript SDK
npm run build
```

### Run a Node

```bash
# Run with default configuration (devnet)
./target/release/ade-node --rpc-port 8899 --data-dir ./data

# Run as validator with custom genesis
./target/release/ade-node \
  --rpc-port 8899 \
  --gossip-port 9900 \
  --data-dir ./data \
  --validator-mode \
  --genesis-config ./genesis.toml \
  --keypair ./validator-keypair.json
```

### Using Docker

```bash
docker build -t ade-node .
docker run -p 8899:8899 -p 9900:9900 -v ./data:/data ade-node
```

## Project Structure

```
ade/
├── node/                   # Core Node Implementation
│   └── src/
│       ├── main.rs         # Entry point
│       ├── validator.rs    # Block production & validation
│       ├── mempool.rs      # Transaction pool with DDoS protection
│       ├── state_transition.rs  # State machine with LRU cache
│       ├── storage.rs      # RocksDB persistent storage
│       ├── network.rs      # Gossipsub P2P networking
│       ├── peer_discovery.rs    # DHT-like peer discovery
│       ├── ai_runtime.rs   # AI agent execution manager
│       ├── ai_inference.rs # ONNX inference engine
│       ├── ai_pytorch.rs   # PyTorch inference (optional)
│       ├── compute_meter.rs    # AI compute cost estimation
│       ├── fee_market.rs   # Dynamic fee adjustment
│       ├── snapshot.rs     # State snapshot management
│       ├── indexer.rs      # Secondary indexes (RocksDB CF)
│       └── block_producer.rs   # Block assembly & priority fees
│
├── consensus/              # Proof-of-Stake Consensus
│   └── src/
│       ├── proof_of_stake.rs   # VRF leader selection, Wesolowski VDF
│       ├── block.rs        # Block validation (skip slot support)
│       ├── vote.rs         # Validator voting
│       └── finality.rs     # Finality gadget
│
├── transaction/            # Transaction Processing
│   └── src/
│       ├── transaction.rs  # Transaction structure
│       ├── instruction.rs  # Instruction parsing
│       ├── executor.rs     # Instruction execution with fee deduction
│       ├── validation.rs   # Signature & structure validation
│       └── account.rs      # Account model
│
├── rpc/                    # RPC Server
│   └── src/
│       ├── server.rs       # HTTP server
│       ├── handlers.rs     # API endpoint handlers
│       ├── websocket.rs    # WebSocket subscriptions
│       ├── state.rs        # RocksDB-backed state
│       └── types.rs        # Request/Response types
│
├── bridge/                 # Cross-Chain Bridge
│   ├── src/
│   │   ├── relayer.rs      # Event polling with pagination
│   │   ├── bridge.rs       # Deposit/withdrawal with per-user nonces
│   │   ├── proof_verification.rs   # Merkle proof verification
│   │   ├── event_parser.rs # Regex-based Solana log parsing
│   │   ├── merkle.rs       # Merkle tree utilities
│   │   └── multisig.rs     # Multi-signature support
│   └── programs/           # Solana Smart Contracts
│       ├── solana-lock/    # Token locking (invoke_signed PDA)
│       └── ade-mint/       # Wrapped token minting (replay protection)
│
├── cli/                    # Command Line Interface
│   └── src/
│       ├── main.rs         # CLI entry point
│       └── commands/       # Subcommands
│           ├── node.rs     # Node management
│           ├── validator.rs    # Validator operations
│           ├── account.rs  # Account queries
│           ├── transaction.rs  # Transaction submission
│           ├── bridge.rs   # Bridge operations
│           └── agent.rs    # AI agent management
│
├── src/                    # TypeScript SDK
│   ├── index.ts            # Main exports
│   ├── client.ts           # RPC client
│   ├── wallet.ts           # HD wallet (BIP-32/39/44)
│   ├── transaction.ts      # Transaction building
│   ├── ai-agent.ts         # AI agent client
│   ├── ai-agent-advanced.ts    # Batch execution, monitoring
│   ├── bridge.ts           # Bridge client
│   ├── bridge-advanced.ts  # Enhanced bridge with history
│   ├── websocket-client.ts # Real-time subscriptions
│   ├── account-manager.ts  # Account operations
│   ├── types/              # TypeScript type definitions
│   │   ├── api.ts          # API types
│   │   └── rpc.ts          # RPC types
│   └── examples/           # Usage examples
│
├── python-sdk/             # Python SDK
│   ├── ade_sidechain/
│   │   ├── client.py       # RPC client
│   │   ├── ai_agent.py     # AI agent operations
│   │   ├── bridge.py       # Bridge operations
│   │   └── transaction.py  # Transaction building
│   └── examples/           # Python examples
│
├── native/                 # C/C++ Native Library
│   ├── src/
│   │   ├── lib.rs          # FFI exports
│   │   ├── crypto.rs       # Ed25519, SHA-256
│   │   └── hash.rs         # Hashing utilities
│   └── ade_native.h        # C header file
│
├── tests/                  # Integration Tests
│   ├── integration_test.rs
│   └── load_test.rs
│
└── docs/                   # Documentation
    ├── architecture.md
    ├── node-setup.md
    ├── rpc-api.md
    ├── ai-agents.md
    ├── bridge.md
    └── consensus.md
```

## Components

### Node

The core validator/full node with production-ready features:

| Component | Description |
|-----------|-------------|
| **Validator** | VRF-based leader selection, target-based slot timing |
| **Mempool** | Balance/nonce verification, DDoS protection |
| **State** | LRU cache with backpressure, lazy loading |
| **Storage** | RocksDB with column families, compression |
| **Network** | Gossipsub mesh, TTL-based message cache |
| **Fee Market** | Dynamic fees with exponential congestion pricing |

### AI Runtime

Deploy and execute AI models on-chain:

```bash
# Supported model formats
- ONNX (recommended, no additional dependencies)
- PyTorch TorchScript (requires libtorch, --features pytorch)
```

| Feature | Description |
|---------|-------------|
| **Compute Metering** | Cost based on model size, params, tokens |
| **Concurrency Control** | Semaphore-limited parallel execution |
| **Async Execution** | spawn_blocking for CPU-intensive inference |
| **Model Cache** | In-memory caching with size limits |

### Bridge

Trustless cross-chain bridge with security features:

| Feature | Description |
|---------|-------------|
| **Merkle Proofs** | Real RPC-fetched proofs, no fallback |
| **Replay Protection** | Processed proof tracking |
| **Per-User Nonces** | Parallel transaction processing |
| **PDA Vaults** | invoke_signed for secure withdrawals |

### Consensus

Proof-of-Stake with grinding-resistant randomness:

| Feature | Description |
|---------|-------------|
| **VRF Leader Selection** | Unpredictable, verifiable selection |
| **Wesolowski VDF** | Time-locked randomness |
| **Skip Slot Support** | Chain continues if validators miss slots |
| **Finality Gadget** | 2-3 second finality |

## SDKs

### TypeScript

```typescript
import { createAdeClient } from 'ade-sidechain';

const client = createAdeClient({
  rpcUrl: 'http://localhost:8899',
  commitment: 'confirmed',
});

// Deploy AI Agent
const agent = await client.deployAIAgent({
  agentId: 'my-agent',
  modelHash: 'QmXxYyZz...',
  config: {
    modelType: 'transformer',
    maxExecutionTime: 30000,
  },
});

// Execute AI Agent
const result = await client.executeAIAgent({
  agentId: 'my-agent',
  inputData: { prompt: 'Hello, AI!' },
  maxCompute: 100000,
});

// Bridge tokens
const deposit = await client.bridgeDeposit({
  amount: 1000000000, // 1 SOL
  targetChain: 'ade',
  recipient: 'AdeAddress...',
});
```

### Python

```python
from ade_sidechain import AdeSidechainClient, AIAgent

client = AdeSidechainClient('http://localhost:8899')

# Deploy and execute AI agent
agent = AIAgent(client)
result = agent.deploy('my-agent', 'QmModelHash...', {
    'model_type': 'transformer',
    'max_tokens': 512
})

execution = agent.execute('my-agent', {'prompt': 'Hello!'})
print(f"Output: {execution['output']}")
```

### CLI

```bash
# Node operations
ade-cli node start --config genesis.toml
ade-cli node status

# Account operations
ade-cli account balance <address>
ade-cli account create --output keypair.json

# AI Agent operations
ade-cli agent deploy --model model.onnx --id my-agent
ade-cli agent execute --id my-agent --input "Hello"

# Bridge operations
ade-cli bridge deposit --amount 1.0 --recipient <ade-address>
ade-cli bridge withdraw --amount 1.0 --recipient <solana-address>

# Validator operations
ade-cli validator stake --amount 1000
ade-cli validator vote --slot 12345
```

## Configuration

### Genesis Configuration (genesis.toml)

```toml
[network]
genesis_hash = "ADE_MAINNET_GENESIS_V1"
network_id = "mainnet"
protocol_version = 1

[node]
keypair_path = "./validator-keypair.json"
gossip_port = 9900
is_validator = true

[bootstrap]
nodes = [
  "node1.ade.network:9900",
  "node2.ade.network:9900",
]

[consensus]
epoch_length = 432000  # ~2 days at 400ms slots
min_stake = 1000000000  # 1 SOL minimum stake

[ai]
max_model_size_mb = 100
max_total_storage_gb = 10
max_concurrent_executions = 4
```

## RPC API

### Chain Methods

| Method | Description |
|--------|-------------|
| `getSlot` | Current slot number |
| `getBlockHeight` | Current block height |
| `getBlock` | Block by slot |
| `getTransaction` | Transaction by signature |

### Account Methods

| Method | Description |
|--------|-------------|
| `getBalance` | Account balance |
| `getAccountInfo` | Full account data |
| `getTokenBalance` | SPL token balance |

### AI Agent Methods

| Method | Description |
|--------|-------------|
| `deployAIAgent` | Deploy new AI agent |
| `executeAIAgent` | Execute agent inference |
| `getAIAgentInfo` | Agent metadata |
| `getAIAgentExecutions` | Execution history |

### Bridge Methods

| Method | Description |
|--------|-------------|
| `bridgeDeposit` | Initiate deposit |
| `bridgeWithdraw` | Initiate withdrawal |
| `getBridgeStatus` | Operation status |

See [RPC API Documentation](docs/rpc-api.md) for details.

## Performance

| Metric | Value |
|--------|-------|
| Block Time | 400ms |
| Transactions/Second | 10,000+ |
| Finality | 2-3 seconds |
| AI Inference | <100ms (typical) |
| State Sync | Incremental snapshots |

## Security

- **Cryptography**: Ed25519 signatures, SHA-256 hashing
- **VDF Randomness**: Wesolowski VDF prevents grinding
- **Merkle Proofs**: Real RPC-fetched, no simulation
- **Replay Protection**: Processed proof tracking
- **DDoS Protection**: Balance/nonce checks in mempool
- **Backpressure**: Prevents memory exhaustion

## Development

### Running Tests

```bash
# Rust tests
cargo test --all

# TypeScript tests
npm test

# Python tests
cd python-sdk && python -m pytest

# Integration tests
cargo test --test integration_test

# Load tests
cargo test --test load_test --release
```

### Building with Features

```bash
# Default build (ONNX only)
cargo build --release

# With PyTorch support
export LIBTORCH=/path/to/libtorch
cargo build --release --features pytorch

# For C/C++ integration
cd native && cargo build --release
```

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Node Setup Guide](docs/node-setup.md)
- [RPC API Reference](docs/rpc-api.md)
- [Transaction Format](docs/transactions.md)
- [AI Agent Integration](docs/ai-agents.md)
- [Bridge Protocol](docs/bridge.md)
- [Consensus Mechanism](docs/consensus.md)

## License

MIT License - see [LICENSE](LICENSE) file

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Contact

- Twitter: [@Ade__labs](https://twitter.com/Ade__labs)
- Donation: `92XwFhgXfxCS2rmRK4EXchGMyW2EqFsNV5zoqWNpw3nS`
- CA: `4aaNon9pZX4nYi2x2LSe8ef9aCjAA1p9NkARaKLvpump`
