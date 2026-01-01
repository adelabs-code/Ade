"""Example: Batch execution of AI agent tasks"""

from ade_sidechain import AdeSidechainClient
from ade_sidechain.ai_agent import AIAgentExecutor


def main():
    # Initialize client
    client = AdeSidechainClient('http://localhost:8899')
    
    # Create executor with batching and caching
    executor = AIAgentExecutor(client)
    
    # Prepare batch inputs
    inputs = [
        {'prompt': 'Summarize: AI is transforming industries...', 'task': 'summarize'},
        {'prompt': 'Translate to French: Hello, how are you?', 'task': 'translate'},
        {'prompt': 'Classify sentiment: This product is amazing!', 'task': 'classify'},
        {'prompt': 'Generate code: Create a Python function to sort a list', 'task': 'generate'},
    ]
    
    # Execute batch
    print("Executing batch of AI tasks...")
    results = executor.execute_batch(
        agent_id='my-ai-agent-001',
        inputs=inputs,
        max_compute_per_execution=50000,
    )
    
    # Print results
    for i, result in enumerate(results):
        print(f"\nTask {i+1}:")
        print(f"  Execution ID: {result.execution_id}")
        print(f"  Compute units: {result.compute_units}")
        print(f"  Signature: {result.signature}")
    
    # Execute with caching
    print("\n\nExecuting with cache...")
    cached_result = executor.execute_with_cache(
        agent_id='my-ai-agent-001',
        input_data=inputs[0],
        cache_key='summarize_task_1',
    )
    print(f"First execution: {cached_result.execution_id}")
    
    # Second call uses cache
    cached_result2 = executor.execute_with_cache(
        agent_id='my-ai-agent-001',
        input_data=inputs[0],
        cache_key='summarize_task_1',
    )
    print(f"Second execution (cached): {cached_result2.execution_id}")
    print(f"Cache hit: {cached_result.execution_id == cached_result2.execution_id}")


if __name__ == '__main__':
    main()


