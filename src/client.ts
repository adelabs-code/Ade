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

export interface ClientOptions {
  timeout?: number;
  retryAttempts?: number;
  retryDelay?: number;
  keepAlive?: boolean;
  maxSockets?: number;
}

export class ConnectionManager {
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 5;
  private reconnectDelay: number = 1000;
  private isConnected: boolean = false;

  constructor(private onReconnect?: () => void) {}

  setConnected(connected: boolean): void {
    this.isConnected = connected;
    if (connected) {
      this.reconnectAttempts = 0;
    }
  }

  async attemptReconnect(): Promise<boolean> {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      return false;
    }

    this.reconnectAttempts++;
    await this.delay(this.reconnectDelay * this.reconnectAttempts);

    if (this.onReconnect) {
      this.onReconnect();
    }

    return true;
  }

  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  getStatus(): { connected: boolean; attempts: number } {
    return {
      connected: this.isConnected,
      attempts: this.reconnectAttempts,
    };
  }

  reset(): void {
    this.reconnectAttempts = 0;
    this.isConnected = false;
  }
}

export class AdeSidechainClient {
  private rpcClient: AxiosInstance;
  private requestId: number = 1;
  private connectionManager: ConnectionManager;
  private options: Required<ClientOptions>;

  constructor(rpcUrl: string, options: ClientOptions = {}) {
    this.options = {
      timeout: options.timeout || 30000,
      retryAttempts: options.retryAttempts || 3,
      retryDelay: options.retryDelay || 1000,
      keepAlive: options.keepAlive !== false,
      maxSockets: options.maxSockets || 100,
    };

    this.rpcClient = axios.create({
      baseURL: rpcUrl,
      timeout: this.options.timeout,
      headers: {
        'Content-Type': 'application/json',
        'Connection': this.options.keepAlive ? 'keep-alive' : 'close',
      },
      maxRedirects: 0,
    });

    this.connectionManager = new ConnectionManager(() => {
      console.log('Attempting to reconnect...');
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // Request interceptor
    this.rpcClient.interceptors.request.use(
      (config) => {
        this.connectionManager.setConnected(true);
        return config;
      },
      (error) => {
        this.connectionManager.setConnected(false);
        return Promise.reject(error);
      }
    );

    // Response interceptor with retry logic
    this.rpcClient.interceptors.response.use(
      (response) => {
        this.connectionManager.setConnected(true);
        return response;
      },
      async (error) => {
        const config = error.config;
        
        if (!config || !config._retry) {
          config._retry = 0;
        }

        if (config._retry < this.options.retryAttempts) {
          config._retry++;
          
          this.connectionManager.setConnected(false);
          await this.delay(this.options.retryDelay * config._retry);
          
          const shouldRetry = await this.connectionManager.attemptReconnect();
          if (shouldRetry) {
            return this.rpcClient(config);
          }
        }

        return Promise.reject(error);
      }
    );
  }

  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
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

  // Chain state methods
  async getSlot(): Promise<number> {
    return this.call<number>('getSlot');
  }

  async getBlockHeight(): Promise<number> {
    return this.call<number>('getBlockHeight');
  }

  async getLatestBlockhash(): Promise<{ blockhash: string; lastValidBlockHeight: number }> {
    return this.call('getLatestBlockhash');
  }

  async getBlock(slot: number): Promise<any> {
    return this.call('getBlock', { slot });
  }

  async getBlocks(startSlot: number, endSlot?: number): Promise<number[]> {
    return this.call('getBlocks', { startSlot, endSlot });
  }

  async getBlockTime(slot: number): Promise<number> {
    return this.call('getBlockTime', { slot });
  }

  async getEpochInfo(): Promise<any> {
    return this.call('getEpochInfo');
  }

  // Transaction methods
  async sendTransaction(transaction: string, options?: any): Promise<string> {
    return this.call<string>('sendTransaction', { transaction, options });
  }

  async simulateTransaction(transaction: string): Promise<any> {
    return this.call('simulateTransaction', { transaction });
  }

  async getTransaction(signature: string): Promise<any> {
    return this.call('getTransaction', { signature });
  }

  async getTransactionCount(): Promise<number> {
    return this.call('getTransactionCount');
  }

  async getSignatureStatuses(signatures: string[]): Promise<any> {
    return this.call('getSignatureStatuses', { signatures });
  }

  async getSignaturesForAddress(address: string, options?: any): Promise<any[]> {
    return this.call('getSignaturesForAddress', { address, ...options });
  }

  async confirmTransaction(signature: string, commitment?: string): Promise<boolean> {
    const result = await this.call('getSignatureStatuses', { signatures: [signature] });
    return result?.value?.[0]?.confirmationStatus === 'finalized';
  }

  // Account methods
  async getBalance(address: string): Promise<{ value: number }> {
    return this.call('getBalance', { address });
  }

  async getAccountInfo(address: string): Promise<any> {
    return this.call('getAccountInfo', { address });
  }

  async getMultipleAccounts(addresses: string[]): Promise<any> {
    return this.call('getMultipleAccounts', { addresses });
  }

  async getProgramAccounts(programId: string): Promise<any[]> {
    return this.call('getProgramAccounts', { programId });
  }

  async getTokenAccountsByOwner(owner: string, options?: any): Promise<any> {
    return this.call('getTokenAccountsByOwner', { owner, ...options });
  }

  // AI Agent methods
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

  async getAIAgentInfo(agentId: string): Promise<any> {
    return this.call('getAIAgentInfo', { agentId });
  }

  async updateAIAgent(agentId: string, newConfig: any): Promise<string> {
    return this.call('updateAIAgent', { agentId, newConfig });
  }

  async listAIAgents(owner?: string): Promise<any> {
    return this.call('listAIAgents', owner ? { owner } : undefined);
  }

  // Bridge methods
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

  async getBridgeStatus(id: string): Promise<any> {
    return this.call('getBridgeStatus', { id });
  }

  async estimateBridgeFee(params: any): Promise<any> {
    return this.call('estimateBridgeFee', params);
  }

  // Validator methods
  async getValidators(): Promise<any> {
    return this.call('getValidators');
  }

  async getVoteAccounts(): Promise<any> {
    return this.call('getVoteAccounts');
  }

  async getLeaderSchedule(slot?: number): Promise<any> {
    return this.call('getLeaderSchedule', slot ? { slot } : undefined);
  }

  // Utility methods
  async requestAirdrop(address: string, lamports: number): Promise<string> {
    return this.call('requestAirdrop', { address, lamports });
  }

  async getHealth(): Promise<string> {
    const response = await axios.get(`${this.rpcClient.defaults.baseURL}/health`);
    return response.data;
  }

  async getMetrics(): Promise<any> {
    const response = await axios.get(`${this.rpcClient.defaults.baseURL}/metrics`);
    return response.data;
  }

  // Connection management
  getConnectionStatus(): { connected: boolean; attempts: number } {
    return this.connectionManager.getStatus();
  }

  resetConnection(): void {
    this.connectionManager.reset();
  }
}
