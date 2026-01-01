# Ade Sidechain Python SDK

Python SDK for interacting with the Ade Sidechain, designed for AI agent integration.

## Installation

```bash
pip install ade-sidechain
```

## Quick Start

```python
from ade_sidechain import AdeSidechainClient, AIAgent, AIAgentConfig

# Initialize client
client = AdeSidechainClient('http://localhost:8899')

# Create AI agent
ai_agent = AIAgent(client)

# Deploy an agent
config = AIAgentConfig(
    model_type='transformer',
    parameters={'max_tokens': 512},
    max_execution_time=30000,
    allowed_operations=['inference'],
)

result = ai_agent.deploy(
    agent_id='my-agent',
    model_hash='QmXxYyZz...',
    config=config,
)

# Execute the agent
execution = ai_agent.execute(
    agent_id='my-agent',
    input_data={'prompt': 'Hello, AI!'},
    max_compute=50000,
)

print(f"Execution ID: {execution.execution_id}")
```

## Features

- Full RPC client implementation
- AI agent deployment and execution
- Bridge operations for cross-chain transfers
- Transaction building utilities
- Batch execution support
- Execution caching

## Examples

See the `examples/` directory for more examples:
- `deploy_ai_agent.py` - Deploy and execute an AI agent
- `bridge_transfer.py` - Bridge assets between chains
- `batch_execution.py` - Batch execution with caching


