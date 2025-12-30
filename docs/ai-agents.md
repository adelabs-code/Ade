# AI Agent Integration Guide

## Overview

Ade Sidechain provides native support for deploying and executing AI agents on-chain. This enables decentralized AI applications with verifiable execution.

## Agent Lifecycle

```
Deploy → Configure → Execute → Monitor → Update
```

## Deployment

### Agent Configuration

```typescript
interface AIAgentConfig {
  modelType: string;              // Model architecture type
  parameters: Record<string, any>; // Model-specific parameters
  maxExecutionTime: number;        // Timeout in milliseconds
  allowedOperations: string[];     // Permitted operations
}
```

### Deployment Example

**TypeScript:**
```typescript
import { AdeSidechainClient, AIAgentClient, AIAgentConfig } from 'ade-sidechain';

const client = new AdeSidechainClient('http://localhost:8899');
const aiAgent = new AIAgentClient(client);

const config: AIAgentConfig = {
  modelType: 'transformer',
  parameters: {
    maxTokens: 512,
    temperature: 0.7,
    topP: 0.9,
  },
  maxExecutionTime: 30000,
  allowedOperations: ['inference', 'embeddings'],
};

const result = await aiAgent.deploy(
  'my-agent-id',
  'QmModelHash...',
  config
);
```

**Python:**
```python
from ade_sidechain import AdeSidechainClient, AIAgent, AIAgentConfig

client = AdeSidechainClient('http://localhost:8899')
ai_agent = AIAgent(client)

config = AIAgentConfig(
    model_type='transformer',
    parameters={'max_tokens': 512, 'temperature': 0.7},
    max_execution_time=30000,
    allowed_operations=['inference']
)

result = ai_agent.deploy('my-agent-id', 'QmModelHash...', config)
```

## Execution

### Single Execution

```python
execution_result = ai_agent.execute(
    agent_id='my-agent-id',
    input_data={
        'prompt': 'Explain quantum computing',
        'max_length': 200
    },
    max_compute=100000
)

print(f"Execution ID: {execution_result.execution_id}")
print(f"Compute used: {execution_result.compute_units}")
```

### Batch Execution

```python
from ade_sidechain.ai_agent import AIAgentExecutor

executor = AIAgentExecutor(client)

inputs = [
    {'prompt': 'Task 1', 'task': 'summarize'},
    {'prompt': 'Task 2', 'task': 'translate'},
    {'prompt': 'Task 3', 'task': 'classify'},
]

results = executor.execute_batch(
    agent_id='my-agent-id',
    inputs=inputs,
    max_compute_per_execution=50000
)
```

### Cached Execution

```python
result = executor.execute_with_cache(
    agent_id='my-agent-id',
    input_data={'prompt': 'Repeated query'},
    cache_key='query_1',
    max_compute=50000
)
```

## Compute Metering

### Compute Units

Different operations consume different compute units:

| Operation | Base Cost | Per-Token Cost |
|-----------|-----------|----------------|
| Model Load | 10,000 | - |
| Inference | 1,000 | 100 |
| Embeddings | 500 | 50 |
| Fine-tuning | 50,000 | 1,000 |

### Budget Management

```typescript
const COMPUTE_BUDGET = 1_400_000; // Max per transaction

const execution = await aiAgent.execute(
  'agent-id',
  inputData,
  COMPUTE_BUDGET
);
```

### Cost Estimation

```python
def estimate_cost(prompt_tokens: int, max_tokens: int) -> int:
    model_load = 10000
    inference_base = 1000
    token_cost = (prompt_tokens + max_tokens) * 100
    return model_load + inference_base + token_cost

cost = estimate_cost(50, 200)  # 35,000 compute units
```

## Model Storage

### IPFS Integration

Models are stored on IPFS and referenced by hash:

```bash
# Upload model to IPFS
ipfs add model.onnx
# Returns: QmXxYyZz1234567890abcdef
```

```typescript
await aiAgent.deploy(
  'agent-id',
  'QmXxYyZz1234567890abcdef', // IPFS hash
  config
);
```

### Supported Formats

- **ONNX**: Portable neural network format
- **TensorFlow Lite**: Mobile-optimized models
- **PyTorch JIT**: Torch script format
- **Custom**: Binary format with runtime adapter

## Agent State

### State Structure

```rust
pub struct AIAgentState {
    pub agent_id: Vec<u8>,
    pub model_hash: Vec<u8>,
    pub owner: Pubkey,
    pub execution_count: u64,
    pub total_compute_used: u64,
    pub config: AgentConfig,
}
```

### Querying State

```typescript
const info = await aiAgent.getInfo('my-agent-id');

console.log(`Owner: ${info.owner}`);
console.log(`Executions: ${info.executionCount}`);
console.log(`Total compute: ${info.totalComputeUsed}`);
```

## Advanced Features

### Multi-Agent Workflows

```python
# Deploy multiple specialized agents
agents = {
    'summarizer': ai_agent.deploy('sum-001', 'QmSum...', sum_config),
    'translator': ai_agent.deploy('trans-001', 'QmTrans...', trans_config),
    'classifier': ai_agent.deploy('class-001', 'QmClass...', class_config),
}

# Chain executions
summary = ai_agent.execute('sum-001', {'text': long_text})
translation = ai_agent.execute('trans-001', {'text': summary.output})
classification = ai_agent.execute('class-001', {'text': translation.output})
```

### Agent Composition

```typescript
interface CompositeAgent {
  agents: string[];
  workflow: WorkflowStep[];
}

interface WorkflowStep {
  agentId: string;
  inputMapping: Record<string, string>;
  outputKey: string;
}
```

### On-Chain Verification

Execution results can be verified on-chain:

```rust
pub fn verify_execution(
    execution_id: &[u8],
    expected_output_hash: &[u8],
) -> bool {
    let execution = get_execution(execution_id)?;
    let actual_hash = hash(&execution.output);
    actual_hash == expected_output_hash
}
```

## Security Considerations

### Sandboxing

All agents execute in isolated environments:
- Memory limits enforced
- No network access
- No file system access
- CPU time limits

### Input Validation

```python
def validate_input(input_data: dict) -> bool:
    if len(str(input_data)) > 10_000:
        return False
    if not all(k.isalnum() for k in input_data.keys()):
        return False
    return True
```

### Access Control

```rust
pub fn check_agent_permission(
    agent_id: &[u8],
    caller: &Pubkey,
    operation: Operation,
) -> Result<()> {
    let agent = get_agent(agent_id)?;
    
    match operation {
        Operation::Execute => Ok(()),
        Operation::Update | Operation::Delete => {
            if caller == &agent.owner {
                Ok(())
            } else {
                Err(Error::Unauthorized)
            }
        }
    }
}
```

## Performance Optimization

### Model Optimization

1. **Quantization**: Reduce model size
```bash
python -m onnxruntime.quantization.preprocess \
  --input model.onnx \
  --output model_quantized.onnx
```

2. **Pruning**: Remove unnecessary weights

3. **Knowledge Distillation**: Train smaller model

### Caching Strategies

```python
class AgentCache:
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
    
    def get_or_execute(self, key: str, execute_fn):
        if key in self.cache:
            return self.cache[key]
        
        result = execute_fn()
        
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[key] = result
        return result
```

## Monitoring and Analytics

### Execution Metrics

```python
info = ai_agent.get_info('my-agent-id')

metrics = {
    'total_executions': info.execution_count,
    'total_compute': info.total_compute_used,
    'avg_compute_per_execution': info.total_compute_used / max(info.execution_count, 1),
}
```

### Event Logging

```typescript
interface ExecutionEvent {
  agentId: string;
  executionId: string;
  timestamp: number;
  computeUsed: number;
  success: boolean;
  error?: string;
}
```

## Best Practices

1. **Model Selection**: Use the smallest model that meets requirements
2. **Compute Budgeting**: Set appropriate compute limits
3. **Error Handling**: Implement retry logic for failed executions
4. **Caching**: Cache frequent queries
5. **Monitoring**: Track execution metrics
6. **Testing**: Test agents thoroughly off-chain first
7. **Versioning**: Version your models and configurations
8. **Documentation**: Document agent behavior and limitations

