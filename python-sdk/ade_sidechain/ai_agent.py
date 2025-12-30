"""AI Agent integration for Ade Sidechain"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from .client import AdeSidechainClient


@dataclass
class AIAgentConfig:
    """Configuration for an AI agent"""
    model_type: str
    parameters: Dict[str, Any]
    max_execution_time: int
    allowed_operations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'modelType': self.model_type,
            'parameters': self.parameters,
            'maxExecutionTime': self.max_execution_time,
            'allowedOperations': self.allowed_operations,
        }


@dataclass
class AIAgentInfo:
    """Information about an AI agent"""
    agent_id: str
    model_hash: str
    owner: str
    execution_count: int
    total_compute_used: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AIAgentInfo':
        return cls(
            agent_id=data['agentId'],
            model_hash=data['modelHash'],
            owner=data['owner'],
            execution_count=data['executionCount'],
            total_compute_used=data['totalComputeUsed'],
        )


@dataclass
class ExecutionResult:
    """Result of an AI agent execution"""
    execution_id: str
    signature: str
    compute_units: int
    output: Optional[Any] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionResult':
        return cls(
            execution_id=data['executionId'],
            signature=data['signature'],
            compute_units=data['computeUnits'],
            output=data.get('output'),
        )


class AIAgent:
    """AI Agent client for Ade Sidechain"""

    def __init__(self, client: AdeSidechainClient):
        self.client = client

    def deploy(
        self,
        agent_id: str,
        model_hash: str,
        config: AIAgentConfig
    ) -> Dict[str, str]:
        """Deploy a new AI agent"""
        return self.client.deploy_ai_agent(
            agent_id=agent_id,
            model_hash=model_hash,
            config=config.to_dict(),
        )

    def execute(
        self,
        agent_id: str,
        input_data: Any,
        max_compute: int = 100000
    ) -> ExecutionResult:
        """Execute an AI agent with input data"""
        result = self.client.execute_ai_agent(
            agent_id=agent_id,
            input_data=input_data,
            max_compute=max_compute,
        )
        return ExecutionResult.from_dict(result)

    def get_info(self, agent_id: str) -> AIAgentInfo:
        """Get information about an AI agent"""
        result = self.client.get_ai_agent_info(agent_id)
        return AIAgentInfo.from_dict(result)


class AIAgentExecutor:
    """Advanced AI agent executor with batching and caching"""

    def __init__(self, client: AdeSidechainClient):
        self.agent = AIAgent(client)
        self.execution_cache: Dict[str, ExecutionResult] = {}

    def execute_batch(
        self,
        agent_id: str,
        inputs: List[Any],
        max_compute_per_execution: int = 100000
    ) -> List[ExecutionResult]:
        """Execute multiple inputs on an AI agent"""
        results = []
        for input_data in inputs:
            result = self.agent.execute(
                agent_id=agent_id,
                input_data=input_data,
                max_compute=max_compute_per_execution,
            )
            results.append(result)
        return results

    def execute_with_cache(
        self,
        agent_id: str,
        input_data: Any,
        cache_key: Optional[str] = None,
        max_compute: int = 100000
    ) -> ExecutionResult:
        """Execute with caching support"""
        if cache_key is None:
            cache_key = f"{agent_id}:{hash(str(input_data))}"

        if cache_key in self.execution_cache:
            return self.execution_cache[cache_key]

        result = self.agent.execute(agent_id, input_data, max_compute)
        self.execution_cache[cache_key] = result
        return result

    def clear_cache(self):
        """Clear execution cache"""
        self.execution_cache.clear()

