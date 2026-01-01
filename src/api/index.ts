/**
 * Ade Sidechain API Client
 */

import axios, { AxiosInstance, AxiosError } from 'axios';
import {
  RpcApi,
  RpcMethod,
  RpcRequest,
  RpcResponse,
  AdeError,
  ErrorCode,
  ClientConfig,
  Slot,
  BlockHash,
  Balance,
  AccountInfo,
  TransactionReceipt,
  AIAgentDeployParams,
  AIAgentDeployResponse,
  AIAgentExecuteParams,
  AIAgentExecuteResponse,
  AIAgentInfo,
  BridgeDepositParams,
  BridgeDepositResponse,
  BridgeWithdrawParams,
  BridgeWithdrawResponse,
  PerformanceMetrics,
  HealthStatus,
  ValidatorInfo,
  Signature,
  PublicKey,
} from '../types';

export class AdeApiClient implements RpcApi {
  private client: AxiosInstance;
  private requestId: number = 1;
  private config: Required<ClientConfig>;

  constructor(config: ClientConfig) {
    this.config = {
      rpcUrl: config.rpcUrl,
      wsUrl: config.wsUrl || '',
      timeout: config.timeout || 30000,
      retryAttempts: config.retryAttempts || 3,
      retryDelay: config.retryDelay || 1000,
      commitment: config.commitment || 'confirmed',
    };

    this.client = axios.create({
      baseURL: this.config.rpcUrl,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        const config = error.config;
        if (!config || !(config as any)._retry) {
          (config as any)._retry = 0;
        }

        if ((config as any)._retry < this.config.retryAttempts) {
          (config as any)._retry++;
          await this.delay(this.config.retryDelay);
          return this.client(config);
        }

        return Promise.reject(error);
      }
    );
  }

  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  private async call<T>(method: RpcMethod, params?: any): Promise<T> {
    const request: RpcRequest = {
      jsonrpc: '2.0',
      id: this.requestId++,
      method,
      params,
    };

    try {
      const response = await this.client.post<RpcResponse<T>>('/', request);

      if (response.data.error) {
        throw new AdeError(
          response.data.error.code,
          response.data.error.message,
          response.data.error.data
        );
      }

      return response.data.result!;
    } catch (error) {
      if (error instanceof AdeError) {
        throw error;
      }

      if (axios.isAxiosError(error)) {
        throw new AdeError(
          ErrorCode.InternalError,
          `Network error: ${error.message}`,
          error
        );
      }

      throw new AdeError(
        ErrorCode.InternalError,
        'Unknown error occurred',
        error
      );
    }
  }

  // Chain State Methods
  async getSlot(): Promise<Slot> {
    return this.call<Slot>(RpcMethod.GetSlot);
  }

  async getBlockHeight(): Promise<number> {
    return this.call<number>(RpcMethod.GetBlockHeight);
  }

  async getLatestBlockhash(): Promise<BlockHash> {
    return this.call<BlockHash>(RpcMethod.GetLatestBlockhash);
  }

  async getBlock(slot: Slot): Promise<any> {
    return this.call(RpcMethod.GetBlock, { slot });
  }

  // Account Methods
  async getBalance(address: PublicKey): Promise<Balance> {
    return this.call<Balance>(RpcMethod.GetBalance, { address });
  }

  async getAccountInfo(address: PublicKey): Promise<AccountInfo> {
    return this.call<AccountInfo>(RpcMethod.GetAccountInfo, { address });
  }

  // Transaction Methods
  async sendTransaction(transaction: string): Promise<Signature> {
    return this.call<Signature>(RpcMethod.SendTransaction, { transaction });
  }

  async getTransaction(signature: Signature): Promise<TransactionReceipt> {
    return this.call<TransactionReceipt>(RpcMethod.GetTransaction, { signature });
  }

  async confirmTransaction(signature: Signature): Promise<boolean> {
    return this.call<boolean>(RpcMethod.ConfirmTransaction, { signature });
  }

  // AI Agent Methods
  async deployAIAgent(params: AIAgentDeployParams): Promise<AIAgentDeployResponse> {
    return this.call<AIAgentDeployResponse>(RpcMethod.DeployAIAgent, params);
  }

  async executeAIAgent(params: AIAgentExecuteParams): Promise<AIAgentExecuteResponse> {
    return this.call<AIAgentExecuteResponse>(RpcMethod.ExecuteAIAgent, params);
  }

  async getAIAgentInfo(agentId: string): Promise<AIAgentInfo> {
    return this.call<AIAgentInfo>(RpcMethod.GetAIAgentInfo, { agentId });
  }

  async updateAIAgent(agentId: string, newConfig: any): Promise<Signature> {
    return this.call<Signature>(RpcMethod.UpdateAIAgent, { agentId, newConfig });
  }

  // Bridge Methods
  async bridgeDeposit(params: BridgeDepositParams): Promise<BridgeDepositResponse> {
    return this.call<BridgeDepositResponse>(RpcMethod.BridgeDeposit, params);
  }

  async bridgeWithdraw(params: BridgeWithdrawParams): Promise<BridgeWithdrawResponse> {
    return this.call<BridgeWithdrawResponse>(RpcMethod.BridgeWithdraw, params);
  }

  async getBridgeStatus(id: string): Promise<any> {
    return this.call(RpcMethod.GetBridgeStatus, { id });
  }

  // Network & Monitoring
  async getHealth(): Promise<HealthStatus> {
    return this.call<HealthStatus>(RpcMethod.GetHealth);
  }

  async getMetrics(): Promise<PerformanceMetrics> {
    return this.call<PerformanceMetrics>(RpcMethod.GetMetrics);
  }

  async getValidators(): Promise<ValidatorInfo[]> {
    return this.call<ValidatorInfo[]>(RpcMethod.GetValidators);
  }

  async getClusterNodes(): Promise<any[]> {
    return this.call<any[]>(RpcMethod.GetClusterNodes);
  }
}

// Export singleton instance creator
export function createAdeClient(config: ClientConfig): AdeApiClient {
  return new AdeApiClient(config);
}


