import { AdeSidechainClient } from './client';

export interface AIAgentConfig {
  modelType: string;
  parameters: Record<string, any>;
  maxExecutionTime: number;
  allowedOperations: string[];
}

export interface AIAgentInfo {
  agentId: string;
  modelHash: string;
  owner: string;
  executionCount: number;
  totalComputeUsed: number;
}

export interface ExecutionResult {
  executionId: string;
  signature: string;
  computeUnits: number;
  output?: any;
}

export class AIAgentClient {
  constructor(private client: AdeSidechainClient) {}

  async deploy(
    agentId: string,
    modelHash: string,
    config: AIAgentConfig
  ): Promise<{ agentId: string; signature: string }> {
    return this.client.deployAIAgent({
      agentId,
      modelHash,
      config,
    });
  }

  async execute(
    agentId: string,
    inputData: any,
    maxCompute: number = 100000
  ): Promise<ExecutionResult> {
    return this.client.executeAIAgent({
      agentId,
      inputData,
      maxCompute,
    });
  }

  async getInfo(agentId: string): Promise<AIAgentInfo> {
    return this.client.getAIAgentInfo(agentId);
  }
}



