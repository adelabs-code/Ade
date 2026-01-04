// Core client
export { AdeSidechainClient, ConnectionManager } from './client';
export type { RpcRequest, RpcResponse, ClientOptions } from './client';

// Transaction building
export { Transaction, TransactionBuilder } from './transaction';
export { AdvancedTransactionBuilder, BatchTransactionBuilder } from './transaction-builder';
export type { Message, MessageHeader, CompiledInstruction, TransactionOptions } from './transaction';

// Instructions
export {
  TransferInstruction,
  AIAgentDeployInstruction,
  AIAgentExecuteInstruction,
  BridgeDepositInstruction,
} from './instruction';
export type { AccountMeta, InstructionType } from './instruction';

// Instruction helpers
export {
  InstructionHelpers,
  SystemProgram,
  ComputeBudgetProgram,
  InstructionValidator,
  FeeCalculator,
} from './instruction-helpers';

// Account management
export { AccountManager, ProgramAccountManager, TokenAccountManager } from './account-manager';
export type { Account, TokenAccount, AccountChangeCallback } from './account-manager';

// Bridge
export { BridgeClient, AdvancedBridgeClient } from './bridge';
export {
  EnhancedBridgeClient,
  BridgeTransactionTracker,
  BridgeFeeEstimator,
  BridgeAnalytics,
} from './bridge-advanced';
export { BridgeStatus } from './bridge';
export type {
  DepositInfo,
  WithdrawalInfo,
  BridgeEvent,
  BridgeClientOptions,
  BridgeTransaction,
  FeeEstimate,
} from './bridge-advanced';

// AI Agent
export { AIAgentClient } from './ai-agent';
export {
  AdvancedAIAgentClient,
  AgentExecutionQueue,
  ModelRegistry,
  ExecutionCache,
} from './ai-agent-advanced';
export type {
  AIAgentConfig,
  AIAgentInfo,
  ExecutionResult as AgentExecutionResult,
  ExecutionTask,
  ModelInfo,
} from './ai-agent-advanced';

// Wallet
export { Wallet, HDWallet, MultiSigWallet, KeypairUtils, AddressUtils } from './wallet';

// WebSocket
export { WebSocketClient, SubscriptionManager } from './websocket-client';
export type {
  SubscriptionOptions,
  SlotInfo,
  AccountNotification,
  SignatureNotification,
  BlockNotification,
  SubscriptionType,
} from './websocket-client';

// API
export { AdeApiClient, createAdeClient } from './api';

// Types
export * from './types';
