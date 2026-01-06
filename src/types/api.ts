/**
 * Ade Sidechain API Type Definitions
 */

// Core Types
export type PublicKey = string;
export type Signature = string;
export type Hash = string;
export type Slot = number;
export type Lamports = number;

// RPC Request/Response Types
export interface RpcRequest<T = any> {
  jsonrpc: '2.0';
  id: number;
  method: string;
  params?: T;
}

export interface RpcResponse<T = any> {
  jsonrpc: '2.0';
  id: number;
  result?: T;
  error?: RpcError;
}

export interface RpcError {
  code: number;
  message: string;
  data?: any;
}

// Chain State Types
export interface BlockInfo {
  slot: Slot;
  parentHash: Hash;
  transactionsRoot: Hash;
  timestamp: number;
  validator: PublicKey;
  transactions: Transaction[];
}

export interface BlockHash {
  blockhash: Hash;
  lastValidBlockHeight: number;
}

export interface AccountInfo {
  lamports: Lamports;
  owner: PublicKey;
  executable: boolean;
  rentEpoch: number;
  data: string | Buffer;
}

export interface Balance {
  value: Lamports;
}

// Transaction Types
export interface Transaction {
  signatures: Uint8Array[];
  message: Message;
}

export interface Message {
  header: MessageHeader;
  accountKeys: Uint8Array[];
  recentBlockhash: Uint8Array;
  instructions: Instruction[];
}

export interface MessageHeader {
  numRequiredSignatures: number;
  numReadonlySignedAccounts: number;
  numReadonlyUnsignedAccounts: number;
}

export interface Instruction {
  programId: PublicKey;
  accounts: AccountMeta[];
  data: Uint8Array;
}

export interface AccountMeta {
  pubkey: PublicKey;
  isSigner: boolean;
  isWritable: boolean;
}

export interface TransactionReceipt {
  slot: Slot;
  transaction: Transaction;
  meta: TransactionMeta;
}

export interface TransactionMeta {
  err: any | null;
  status: { Ok: null } | { Err: string };
  fee?: Lamports;
  preBalances?: Lamports[];
  postBalances?: Lamports[];
  logMessages?: string[];
  computeUnitsConsumed?: number;
}

// AI Agent Types
export interface AIAgentConfig {
  modelType: string;
  parameters: Record<string, any>;
  maxExecutionTime: number;
  allowedOperations: string[];
}

export interface AIAgentDeployParams {
  agentId: string;
  modelHash: string;
  config: AIAgentConfig;
}

export interface AIAgentDeployResponse {
  agentId: string;
  signature: Signature;
}

export interface AIAgentExecuteParams {
  agentId: string;
  inputData: any;
  maxCompute: number;
}

export interface AIAgentExecuteResponse {
  executionId: string;
  signature: Signature;
  computeUnits: number;
  output?: any;
}

export interface AIAgentInfo {
  agentId: string;
  modelHash: string;
  owner: PublicKey;
  executionCount: number;
  totalComputeUsed: number;
}

// Bridge Types
export enum BridgeStatus {
  Pending = 'pending',
  Locked = 'locked',
  Relayed = 'relayed',
  Completed = 'completed',
  Failed = 'failed',
}

export interface BridgeDepositParams {
  fromChain: string;
  amount: Lamports;
  tokenAddress: PublicKey;
}

export interface BridgeDepositResponse {
  depositId: string;
  signature: Signature;
}

export interface BridgeWithdrawParams {
  toChain: string;
  amount: Lamports;
  recipient: PublicKey;
}

export interface BridgeWithdrawResponse {
  withdrawalId: string;
  signature: Signature;
}

export interface DepositInfo {
  depositId: string;
  fromChain: string;
  toChain: string;
  amount: Lamports;
  token: PublicKey;
  sender: PublicKey;
  recipient: PublicKey;
  status: BridgeStatus;
  timestamp: number;
}

export interface WithdrawalInfo {
  withdrawalId: string;
  fromChain: string;
  toChain: string;
  amount: Lamports;
  token: PublicKey;
  sender: PublicKey;
  recipient: PublicKey;
  status: BridgeStatus;
  timestamp: number;
}

// Validator & Consensus Types
export interface ValidatorInfo {
  pubkey: PublicKey;
  stake: Lamports;
  commission: number;
  lastVoteSlot: Slot;
  active: boolean;
}

export interface VoteState {
  votes: Vote[];
  rootSlot: Slot;
  lastTimestamp: number;
}

export interface Vote {
  slot: Slot;
  blockHash: Hash;
  validator: PublicKey;
  timestamp: number;
}

// Network & Performance Types
export interface ClusterNode {
  pubkey: PublicKey;
  gossip: string;
  rpc: string;
  version: string;
}

export interface PerformanceMetrics {
  slot: Slot;
  transactionCount: number;
  tps: number;
  averageBlockTime: number;
  validatorCount: number;
}

export interface HealthStatus {
  status: 'ok' | 'degraded' | 'down';
  slot: Slot;
  behindBy?: number;
}

// Event Types
export interface TransactionEvent {
  signature: Signature;
  slot: Slot;
  err: any | null;
}

export interface SlotEvent {
  slot: Slot;
  parent: Slot;
  timestamp: number;
}

export interface AIAgentEvent {
  type: 'deploy' | 'execute' | 'update';
  agentId: string;
  signature: Signature;
  slot: Slot;
}

export interface BridgeEvent {
  type: 'deposit' | 'withdraw';
  id: string;
  signature: Signature;
  slot: Slot;
  status: BridgeStatus;
}

// API Method Parameter Types
export interface GetTransactionParams {
  signature: Signature;
  commitment?: Commitment;
}

export interface GetAccountInfoParams {
  address: PublicKey;
  commitment?: Commitment;
}

export interface GetBalanceParams {
  address: PublicKey;
  commitment?: Commitment;
}

export interface SendTransactionParams {
  transaction: string;
  options?: {
    skipPreflight?: boolean;
    preflightCommitment?: Commitment;
  };
}

// Commitment Levels
export type Commitment = 'processed' | 'confirmed' | 'finalized';

export interface CommitmentConfig {
  commitment?: Commitment;
}

// Pagination
export interface PaginationParams {
  limit?: number;
  offset?: number;
  before?: string;
  after?: string;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  hasMore: boolean;
  nextCursor?: string;
}

// Error Types
export enum ErrorCode {
  InvalidRequest = -32600,
  MethodNotFound = -32601,
  InvalidParams = -32602,
  InternalError = -32603,
  ParseError = -32700,
  
  // Custom error codes
  InsufficientBalance = -32001,
  InvalidSignature = -32002,
  TransactionTooLarge = -32003,
  ComputeBudgetExceeded = -32004,
  BlockhashExpired = -32005,
  AccountNotFound = -32006,
  AgentNotFound = -32007,
}

export class AdeError extends Error {
  constructor(
    public code: ErrorCode,
    message: string,
    public data?: any
  ) {
    super(message);
    this.name = 'AdeError';
  }
}

// Configuration Types
export interface ClientConfig {
  rpcUrl: string;
  wsUrl?: string;
  timeout?: number;
  retryAttempts?: number;
  retryDelay?: number;
  commitment?: Commitment;
}

export interface TransactionOptions {
  skipPreflight?: boolean;
  preflightCommitment?: Commitment;
  maxRetries?: number;
}

// Utility Types
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export type RequireAtLeastOne<T, Keys extends keyof T = keyof T> = 
  Pick<T, Exclude<keyof T, Keys>> & 
  {
    [K in Keys]-?: Required<Pick<T, K>> & Partial<Pick<T, Exclude<Keys, K>>>
  }[Keys];

export type Awaitable<T> = T | Promise<T>;








