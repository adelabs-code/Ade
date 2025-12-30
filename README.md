# Ade Sidechain

<div align="center">
  <img src="docs/maina.png" alt="Ade Sidechain" width="400"/>
  
  [![Twitter Follow](https://img.shields.io/twitter/follow/Ade__labs?style=social)](https://twitter.com/Ade__labs)
</div>

A Solana-based sidechain optimized for AI agent deployment and execution.

## Overview

Ade is a high-performance sidechain that extends Solana's capabilities to provide specialized infrastructure for AI agents. It features:

- **AI Agent Runtime**: Deploy and execute AI models on-chain with compute metering
- **Cross-chain Bridge**: Seamless asset transfer between Solana and Ade
- **Proof-of-Stake Consensus**: Energy-efficient validator selection
- **High Throughput**: Low-latency transaction processing optimized for AI workloads
- **Multi-language SDKs**: TypeScript, Python, and Rust support

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Ade Sidechain                      │
├─────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │   Node   │  │   RPC    │  │  Bridge  │          │
│  │  Layer   │  │  Server  │  │  Relayer │          │
│  └──────────┘  └──────────┘  └──────────┘          │
├─────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │   PoS    │  │   Tx     │  │ AI Agent │          │
│  │Consensus │  │Validator │  │ Runtime  │          │
│  └──────────┘  └──────────┘  └──────────┘          │
├─────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ Storage  │  │ Network  │  │  Native  │          │
│  │(RocksDB) │  │ (Gossip) │  │  Crypto  │          │
│  └──────────┘  └──────────┘  └──────────┘          │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### Build from Source

```bash
# Install dependencies
make install-deps

# Build all components
make build

# Run node
./target/release/ade-node \
  --rpc-port 8899 \
  --data-dir ./data
```

### Using Docker

```bash
docker build -t ade-node .
docker run -p 8899:8899 ade-node
```

## Components

### Node

The core validator/full node implementation.

**Key Features:**
- Block production and validation
- Transaction processing
- State management with RocksDB
- P2P networking via gossip protocol

**Configuration:**
```bash
ade-node \
  --rpc-port 8899 \
  --gossip-port 9900 \
  --data-dir ./data \
  --validator-mode \
  --validator-keypair ./keypair.json
```

### RPC Server

JSON-RPC 2.0 API for client interaction.

**Endpoints:**
- `getSlot` - Get current slot
- `getBlockHeight` - Get block height
- `sendTransaction` - Submit transaction
- `deployAIAgent` - Deploy AI agent
- `executeAIAgent` - Execute AI agent
- `bridgeDeposit` - Initiate bridge deposit
- `bridgeWithdraw` - Initiate bridge withdrawal

See [RPC API Documentation](docs/rpc-api.md) for details.

### Bridge

Cross-chain bridge for asset transfers between Solana and Ade.

**Flow:**
1. User locks assets on source chain
2. Relayer detects lock event
3. Relayer submits proof to destination chain
4. Assets minted/unlocked on destination

See [Bridge Documentation](docs/bridge.md) for details.

## SDKs

### TypeScript

```typescript
import { AdeSidechainClient, AIAgentClient } from 'ade-sidechain';

const client = new AdeSidechainClient('http://localhost:8899');
const aiAgent = new AIAgentClient(client);

const result = await aiAgent.deploy(
  'agent-id',
  'model-hash',
  config
);
```

### Python

```python
from ade_sidechain import AdeSidechainClient, AIAgent

client = AdeSidechainClient('http://localhost:8899')
ai_agent = AIAgent(client)

result = ai_agent.deploy('agent-id', 'model-hash', config)
```

### Rust

```rust
use ade_transaction::{Transaction, TransactionBuilder};

let tx = TransactionBuilder::new()
    .add_instruction(instruction)
    .set_recent_blockhash(blockhash)
    .build()?;
```

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Node Setup Guide](docs/node-setup.md)
- [RPC API Reference](docs/rpc-api.md)
- [Transaction Format](docs/transactions.md)
- [AI Agent Integration](docs/ai-agents.md)
- [Bridge Protocol](docs/bridge.md)
- [Consensus Mechanism](docs/consensus.md)

## Development

### Running Tests

```bash
# Rust tests
cargo test --all

# TypeScript tests
npm test

# Python tests
cd python-sdk && python -m pytest
```

### Project Structure

```
ade/
├── node/           # Node implementation
├── transaction/    # Transaction types and validation
├── consensus/      # Proof-of-Stake consensus
├── rpc/            # RPC server
├── bridge/         # Bridge components
├── native/         # Native C library
├── src/            # TypeScript SDK
├── python-sdk/     # Python SDK
└── docs/           # Documentation
```

## Performance

- **Block Time**: 400ms
- **TPS**: 10,000+ transactions per second
- **Finality**: 2-3 seconds
- **AI Agent Execution**: <100ms for inference tasks

## Security

- Ed25519 signatures
- SHA-256 hashing
- Hardware wallet support (planned)
- Multi-signature transactions (planned)

## License

MIT License - see [LICENSE](LICENSE) file

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Contact

- Twitter: [@Ade__labs](https://twitter.com/Ade__labs)

