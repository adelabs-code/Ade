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
  // Block Methods
  getSlot(): Promise<Slot>;
  getBlockHeight(): Promise<number>;
  getBlock(slot: Slot): Promise<any>;
  getBlocks(startSlot: Slot, endSlot?: Slot): Promise<Slot[]>;
  getBlockTime(slot: Slot): Promise<number>;
  getFirstAvailableBlock(): Promise<Slot>;
  getLatestBlockhash(): Promise<BlockHash>;
  getBlockProduction(): Promise<any>;
  getBlockCommitment(slot: Slot): Promise<any>;
  
  // Account Methods
  getBalance(address: PublicKey): Promise<Balance>;
  getAccountInfo(address: PublicKey): Promise<AccountInfo>;
  getMultipleAccounts(addresses: PublicKey[]): Promise<AccountInfo[]>;
  getProgramAccounts(programId: PublicKey): Promise<any[]>;
  getLargestAccounts(): Promise<any[]>;
  getTokenAccountsByOwner(owner: PublicKey, mint?: PublicKey): Promise<any[]>;
  getTokenSupply(mint: PublicKey): Promise<any>;
  
  // Transaction Methods
  sendTransaction(transaction: string): Promise<Signature>;
  simulateTransaction(transaction: string): Promise<any>;
  getTransaction(signature: Signature): Promise<TransactionReceipt>;
  getTransactionCount(): Promise<number>;
  getRecentPerformanceSamples(limit?: number): Promise<any[]>;
  getSignatureStatuses(signatures: Signature[]): Promise<any>;
  getSignaturesForAddress(address: PublicKey, options?: any): Promise<any[]>;
  confirmTransaction(signature: Signature): Promise<boolean>;
  
  // Validator & Staking Methods
  getVoteAccounts(): Promise<any>;
  getValidators(): Promise<ValidatorInfo[]>;
  getStakeActivation(stakeAccount: PublicKey): Promise<any>;
  getStakeMinimumDelegation(): Promise<number>;
  getLeaderSchedule(slot?: Slot): Promise<Record<string, number[]>>;
  getEpochInfo(): Promise<any>;
  getEpochSchedule(): Promise<any>;
  
  // Network & Cluster Methods
  getClusterNodes(): Promise<any[]>;
  getVersion(): Promise<any>;
  getGenesisHash(): Promise<string>;
  getIdentity(): Promise<any>;
  getInflationGovernor(): Promise<any>;
  getInflationRate(): Promise<any>;
  getInflationReward(addresses: PublicKey[], epoch?: number): Promise<any[]>;
  getSupply(): Promise<any>;
  
  // AI Agent Methods
  deployAIAgent(params: AIAgentDeployParams): Promise<AIAgentDeployResponse>;
  executeAIAgent(params: AIAgentExecuteParams): Promise<AIAgentExecuteResponse>;
  getAIAgentInfo(agentId: string): Promise<AIAgentInfo>;
  updateAIAgent(agentId: string, newConfig: any): Promise<Signature>;
  listAIAgents(owner?: PublicKey): Promise<any>;
  getAIAgentExecutions(agentId: string): Promise<any>;
  
  // Bridge Methods
  bridgeDeposit(params: BridgeDepositParams): Promise<BridgeDepositResponse>;
  bridgeWithdraw(params: BridgeWithdrawParams): Promise<BridgeWithdrawResponse>;
  getBridgeStatus(id: string): Promise<any>;
  getBridgeHistory(address?: PublicKey): Promise<any>;
  estimateBridgeFee(params: any): Promise<any>;
  
  // Utility Methods
  requestAirdrop(address: PublicKey, lamports: number): Promise<Signature>;
  minimumLedgerSlot(): Promise<Slot>;
  getSlotLeaders(start: Slot, limit: number): Promise<PublicKey[]>;
  getFeeForMessage(message: string): Promise<number>;
  getRecentPrioritizationFees(): Promise<any[]>;
  getMaxRetransmitSlot(): Promise<Slot>;
  getMaxShredInsertSlot(): Promise<Slot>;
  
  // Network & Monitoring
  getHealth(): Promise<HealthStatus>;
  getMetrics(): Promise<PerformanceMetrics>;
}

/**
 * RPC Method Names
 */
export enum RpcMethod {
  // Block Methods
  GetSlot = 'getSlot',
  GetBlockHeight = 'getBlockHeight',
  GetBlock = 'getBlock',
  GetBlocks = 'getBlocks',
  GetBlockTime = 'getBlockTime',
  GetFirstAvailableBlock = 'getFirstAvailableBlock',
  GetLatestBlockhash = 'getLatestBlockhash',
  GetBlockProduction = 'getBlockProduction',
  GetBlockCommitment = 'getBlockCommitment',
  
  // Account Methods
  GetBalance = 'getBalance',
  GetAccountInfo = 'getAccountInfo',
  GetMultipleAccounts = 'getMultipleAccounts',
  GetProgramAccounts = 'getProgramAccounts',
  GetLargestAccounts = 'getLargestAccounts',
  GetTokenAccountsByOwner = 'getTokenAccountsByOwner',
  GetTokenSupply = 'getTokenSupply',
  
  // Transaction Methods
  SendTransaction = 'sendTransaction',
  SimulateTransaction = 'simulateTransaction',
  GetTransaction = 'getTransaction',
  GetTransactionCount = 'getTransactionCount',
  GetRecentPerformanceSamples = 'getRecentPerformanceSamples',
  GetSignatureStatuses = 'getSignatureStatuses',
  GetSignaturesForAddress = 'getSignaturesForAddress',
  ConfirmTransaction = 'confirmTransaction',
  
  // Validator & Staking Methods
  GetVoteAccounts = 'getVoteAccounts',
  GetValidators = 'getValidators',
  GetStakeActivation = 'getStakeActivation',
  GetStakeMinimumDelegation = 'getStakeMinimumDelegation',
  GetLeaderSchedule = 'getLeaderSchedule',
  GetEpochInfo = 'getEpochInfo',
  GetEpochSchedule = 'getEpochSchedule',
  
  // Network & Cluster Methods
  GetClusterNodes = 'getClusterNodes',
  GetVersion = 'getVersion',
  GetGenesisHash = 'getGenesisHash',
  GetIdentity = 'getIdentity',
  GetInflationGovernor = 'getInflationGovernor',
  GetInflationRate = 'getInflationRate',
  GetInflationReward = 'getInflationReward',
  GetSupply = 'getSupply',
  
  // AI Agent Methods
  DeployAIAgent = 'deployAIAgent',
  ExecuteAIAgent = 'executeAIAgent',
  GetAIAgentInfo = 'getAIAgentInfo',
  UpdateAIAgent = 'updateAIAgent',
  ListAIAgents = 'listAIAgents',
  GetAIAgentExecutions = 'getAIAgentExecutions',
  
  // Bridge Methods
  BridgeDeposit = 'bridgeDeposit',
  BridgeWithdraw = 'bridgeWithdraw',
  GetBridgeStatus = 'getBridgeStatus',
  GetBridgeHistory = 'getBridgeHistory',
  EstimateBridgeFee = 'estimateBridgeFee',
  
  // Utility Methods
  RequestAirdrop = 'requestAirdrop',
  MinimumLedgerSlot = 'minimumLedgerSlot',
  GetSlotLeaders = 'getSlotLeaders',
  GetFeeForMessage = 'getFeeForMessage',
  GetRecentPrioritizationFees = 'getRecentPrioritizationFees',
  GetMaxRetransmitSlot = 'getMaxRetransmitSlot',
  GetMaxShredInsertSlot = 'getMaxShredInsertSlot',
  
  // Network & Monitoring
  GetHealth = 'getHealth',
  GetMetrics = 'getMetrics',
}

/**
 * Typed RPC Request Builder
 */
export interface TypedRpcRequest {
  // Block Methods
  [RpcMethod.GetSlot]: {
    params?: void;
    result: Slot;
  };
  [RpcMethod.GetBlockHeight]: {
    params?: void;
    result: number;
  };
  [RpcMethod.GetBlock]: {
    params: { slot: Slot };
    result: any;
  };
  [RpcMethod.GetBlocks]: {
    params: { startSlot: Slot; endSlot?: Slot };
    result: Slot[];
  };
  
  // Account Methods
  [RpcMethod.GetBalance]: {
    params: { address: PublicKey };
    result: Balance;
  };
  [RpcMethod.GetAccountInfo]: {
    params: { address: PublicKey };
    result: AccountInfo;
  };
  [RpcMethod.GetMultipleAccounts]: {
    params: { addresses: PublicKey[] };
    result: AccountInfo[];
  };
  
  // Transaction Methods
  [RpcMethod.SendTransaction]: {
    params: { transaction: string };
    result: Signature;
  };
  [RpcMethod.GetTransaction]: {
    params: { signature: Signature };
    result: TransactionReceipt;
  };
  [RpcMethod.GetTransactionCount]: {
    params?: void;
    result: number;
  };
  
  // AI Agent Methods
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
  [RpcMethod.ListAIAgents]: {
    params?: { owner?: PublicKey };
    result: any;
  };
  
  // Bridge Methods
  [RpcMethod.BridgeDeposit]: {
    params: BridgeDepositParams;
    result: BridgeDepositResponse;
  };
  [RpcMethod.BridgeWithdraw]: {
    params: BridgeWithdrawParams;
    result: BridgeWithdrawResponse;
  };
  [RpcMethod.GetBridgeStatus]: {
    params: { id: string };
    result: any;
  };
  
  // Utility Methods
  [RpcMethod.RequestAirdrop]: {
    params: { address: PublicKey; lamports: number };
    result: Signature;
  };
  [RpcMethod.GetSlotLeaders]: {
    params: { start: Slot; limit: number };
    result: PublicKey[];
  };
}
