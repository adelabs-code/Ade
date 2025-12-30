/**
 * RPC Method Interface Definitions
 */

import {
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
} from './api';

/**
 * Complete RPC API Interface
 */
export interface RpcApi {
  // Chain State Methods
  getSlot(): Promise<Slot>;
  getBlockHeight(): Promise<number>;
  getLatestBlockhash(): Promise<BlockHash>;
  getBlock(slot: Slot): Promise<any>;
  
  // Account Methods
  getBalance(address: PublicKey): Promise<Balance>;
  getAccountInfo(address: PublicKey): Promise<AccountInfo>;
  
  // Transaction Methods
  sendTransaction(transaction: string): Promise<Signature>;
  getTransaction(signature: Signature): Promise<TransactionReceipt>;
  confirmTransaction(signature: Signature): Promise<boolean>;
  
  // AI Agent Methods
  deployAIAgent(params: AIAgentDeployParams): Promise<AIAgentDeployResponse>;
  executeAIAgent(params: AIAgentExecuteParams): Promise<AIAgentExecuteResponse>;
  getAIAgentInfo(agentId: string): Promise<AIAgentInfo>;
  updateAIAgent(agentId: string, newConfig: any): Promise<Signature>;
  
  // Bridge Methods
  bridgeDeposit(params: BridgeDepositParams): Promise<BridgeDepositResponse>;
  bridgeWithdraw(params: BridgeWithdrawParams): Promise<BridgeWithdrawResponse>;
  getBridgeStatus(id: string): Promise<any>;
  
  // Network & Monitoring
  getHealth(): Promise<HealthStatus>;
  getMetrics(): Promise<PerformanceMetrics>;
  getValidators(): Promise<ValidatorInfo[]>;
  getClusterNodes(): Promise<any[]>;
}

/**
 * RPC Method Names
 */
export enum RpcMethod {
  // Chain
  GetSlot = 'getSlot',
  GetBlockHeight = 'getBlockHeight',
  GetLatestBlockhash = 'getLatestBlockhash',
  GetBlock = 'getBlock',
  
  // Accounts
  GetBalance = 'getBalance',
  GetAccountInfo = 'getAccountInfo',
  
  // Transactions
  SendTransaction = 'sendTransaction',
  GetTransaction = 'getTransaction',
  ConfirmTransaction = 'confirmTransaction',
  
  // AI Agents
  DeployAIAgent = 'deployAIAgent',
  ExecuteAIAgent = 'executeAIAgent',
  GetAIAgentInfo = 'getAIAgentInfo',
  UpdateAIAgent = 'updateAIAgent',
  
  // Bridge
  BridgeDeposit = 'bridgeDeposit',
  BridgeWithdraw = 'bridgeWithdraw',
  GetBridgeStatus = 'getBridgeStatus',
  
  // Network
  GetHealth = 'getHealth',
  GetMetrics = 'getMetrics',
  GetValidators = 'getValidators',
  GetClusterNodes = 'getClusterNodes',
}

/**
 * Typed RPC Request Builder
 */
export interface TypedRpcRequest {
  // Chain
  [RpcMethod.GetSlot]: {
    params?: void;
    result: Slot;
  };
  [RpcMethod.GetBlockHeight]: {
    params?: void;
    result: number;
  };
  [RpcMethod.GetLatestBlockhash]: {
    params?: void;
    result: BlockHash;
  };
  
  // Accounts
  [RpcMethod.GetBalance]: {
    params: { address: PublicKey };
    result: Balance;
  };
  [RpcMethod.GetAccountInfo]: {
    params: { address: PublicKey };
    result: AccountInfo;
  };
  
  // Transactions
  [RpcMethod.SendTransaction]: {
    params: { transaction: string };
    result: Signature;
  };
  [RpcMethod.GetTransaction]: {
    params: { signature: Signature };
    result: TransactionReceipt;
  };
  
  // AI Agents
  [RpcMethod.DeployAIAgent]: {
    params: AIAgentDeployParams;
    result: AIAgentDeployResponse;
  };
  [RpcMethod.ExecuteAIAgent]: {
    params: AIAgentExecuteParams;
    result: AIAgentExecuteResponse;
  };
  [RpcMethod.GetAIAgentInfo]: {
    params: { agentId: string };
    result: AIAgentInfo;
  };
  
  // Bridge
  [RpcMethod.BridgeDeposit]: {
    params: BridgeDepositParams;
    result: BridgeDepositResponse;
  };
  [RpcMethod.BridgeWithdraw]: {
    params: BridgeWithdrawParams;
    result: BridgeWithdrawResponse;
  };
}

