"""Example: Deploy and execute an AI agent"""

from ade_sidechain import AdeSidechainClient, AIAgent, AIAgentConfig


def main():
    # Initialize client
    client = AdeSidechainClient('http://localhost:8899')
    
    # Create AI agent client
    ai_agent = AIAgent(client)
    
    # Configure the AI agent
    config = AIAgentConfig(
        model_type='transformer',
        parameters={
            'max_tokens': 512,
            'temperature': 0.7,
            'top_p': 0.9,
        },
        max_execution_time=30000,  # 30 seconds
        allowed_operations=['inference', 'embeddings'],
    )
    
    # Deploy the agent
    print("Deploying AI agent...")
    result = ai_agent.deploy(
        agent_id='my-ai-agent-001',
        model_hash='QmXxYyZz1234567890abcdef',
        config=config,
    )
    
    print(f"Agent deployed!")
    print(f"Agent ID: {result['agentId']}")
    print(f"Transaction signature: {result['signature']}")
    
    # Execute the agent
    print("\nExecuting AI agent...")
    execution_result = ai_agent.execute(
        agent_id='my-ai-agent-001',
        input_data={
            'prompt': 'What is the meaning of life?',
            'max_length': 100,
        },
        max_compute=50000,
    )
    
    print(f"Execution ID: {execution_result.execution_id}")
    print(f"Compute units used: {execution_result.compute_units}")
    print(f"Transaction signature: {execution_result.signature}")
    
    # Get agent info
    print("\nFetching agent info...")
    info = ai_agent.get_info('my-ai-agent-001')
    print(f"Model hash: {info.model_hash}")
    print(f"Owner: {info.owner}")
    print(f"Total executions: {info.execution_count}")
    print(f"Total compute used: {info.total_compute_used}")


if __name__ == '__main__':
    main()

