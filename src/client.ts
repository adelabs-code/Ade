import axios, { AxiosInstance } from 'axios';

export interface RpcRequest {
  jsonrpc: string;
  id: number;
  method: string;
  params?: any;
}

export interface RpcResponse<T = any> {
  jsonrpc: string;
  id: number;
  result?: T;
  error?: {
    code: number;
    message: string;
  };
}

export class AdeSidechainClient {
  private rpcClient: AxiosInstance;
  private requestId: number = 1;

  constructor(rpcUrl: string) {
    this.rpcClient = axios.create({
      baseURL: rpcUrl,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  private async call<T>(method: string, params?: any): Promise<T> {
    const request: RpcRequest = {
      jsonrpc: '2.0',
      id: this.requestId++,
      method,
      params,
    };

    const response = await this.rpcClient.post<RpcResponse<T>>('/', request);

    if (response.data.error) {
      throw new Error(`RPC Error: ${response.data.error.message}`);
    }

    return response.data.result!;
  }

  async getSlot(): Promise<number> {
    return this.call<number>('getSlot');
  }

  async getBlockHeight(): Promise<number> {
    return this.call<number>('getBlockHeight');
  }

  async getLatestBlockhash(): Promise<{ blockhash: string; lastValidBlockHeight: number }> {
    return this.call('getLatestBlockhash');
  }

  async sendTransaction(transaction: string): Promise<string> {
    return this.call<string>('sendTransaction', { transaction });
  }

  async getTransaction(signature: string): Promise<any> {
    return this.call('getTransaction', { signature });
  }

  async getBalance(address: string): Promise<{ value: number }> {
    return this.call('getBalance', { address });
  }

  async getAccountInfo(address: string): Promise<any> {
    return this.call('getAccountInfo', { address });
  }

  async deployAIAgent(params: {
    agentId: string;
    modelHash: string;
    config: any;
  }): Promise<{ agentId: string; signature: string }> {
    return this.call('deployAIAgent', params);
  }

  async executeAIAgent(params: {
    agentId: string;
    inputData: any;
    maxCompute: number;
  }): Promise<{ executionId: string; signature: string; computeUnits: number }> {
    return this.call('executeAIAgent', params);
  }

  async getAIAgentInfo(agentId: string): Promise<{
    agentId: string;
    modelHash: string;
    owner: string;
    executionCount: number;
    totalComputeUsed: number;
  }> {
    return this.call('getAIAgentInfo', { agentId });
  }

  async bridgeDeposit(params: {
    fromChain: string;
    amount: number;
    tokenAddress: string;
  }): Promise<{ depositId: string; signature: string }> {
    return this.call('bridgeDeposit', params);
  }

  async bridgeWithdraw(params: {
    toChain: string;
    amount: number;
    recipient: string;
  }): Promise<{ withdrawalId: string; signature: string }> {
    return this.call('bridgeWithdraw', params);
  }
}

