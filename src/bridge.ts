import { AdeSidechainClient } from './client';
import { EventEmitter } from 'events';

export enum BridgeStatus {
  Pending = 'pending',
  Locked = 'locked',
  Relayed = 'relayed',
  Completed = 'completed',
  Failed = 'failed',
}

export interface DepositInfo {
  depositId: string;
  fromChain: string;
  toChain: string;
  amount: number;
  token: string;
  sender: string;
  recipient: string;
  status: BridgeStatus;
  timestamp: number;
  confirmations?: number;
  txHash?: string;
}

export interface WithdrawalInfo {
  withdrawalId: string;
  fromChain: string;
  toChain: string;
  amount: number;
  token: string;
  sender: string;
  recipient: string;
  status: BridgeStatus;
  timestamp: number;
  confirmations?: number;
  txHash?: string;
}

export interface BridgeEvent {
  type: 'deposit' | 'withdraw' | 'statusChange';
  id: string;
  status: BridgeStatus;
  data: any;
  timestamp: number;
}

export interface BridgeClientOptions {
  pollInterval?: number;
  confirmationThreshold?: number;
  maxRetries?: number;
  timeout?: number;
}

export class BridgeClient extends EventEmitter {
  private pollInterval: number;
  private confirmationThreshold: number;
  private maxRetries: number;
  private timeout: number;
  private activePolls: Map<string, NodeJS.Timeout> = new Map();

  constructor(
    private client: AdeSidechainClient,
    options: BridgeClientOptions = {}
  ) {
    super();
    this.pollInterval = options.pollInterval || 5000;
    this.confirmationThreshold = options.confirmationThreshold || 32;
    this.maxRetries = options.maxRetries || 3;
    this.timeout = options.timeout || 300000; // 5 minutes
  }

  /**
   * Initiate a deposit from another chain to Ade sidechain
   */
  async deposit(
    fromChain: string,
    amount: number,
    tokenAddress: string,
    recipient?: string
  ): Promise<DepositInfo> {
    const result = await this.client.bridgeDeposit({
      fromChain,
      amount,
      tokenAddress,
    });

    const depositInfo: DepositInfo = {
      depositId: result.depositId,
      fromChain,
      toChain: 'ade',
      amount,
      token: tokenAddress,
      sender: '', // Will be filled by backend
      recipient: recipient || '',
      status: BridgeStatus.Pending,
      timestamp: Date.now(),
      txHash: result.signature,
    };

    // Start monitoring deposit status
    this.startMonitoring(result.depositId, 'deposit');

    return depositInfo;
  }

  /**
   * Initiate a withdrawal from Ade sidechain to another chain
   */
  async withdraw(
    toChain: string,
    amount: number,
    recipient: string,
    tokenAddress?: string
  ): Promise<WithdrawalInfo> {
    const result = await this.client.bridgeWithdraw({
      toChain,
      amount,
      recipient,
    });

    const withdrawalInfo: WithdrawalInfo = {
      withdrawalId: result.withdrawalId,
      fromChain: 'ade',
      toChain,
      amount,
      token: tokenAddress || '',
      sender: '', // Will be filled by backend
      recipient,
      status: BridgeStatus.Pending,
      timestamp: Date.now(),
      txHash: result.signature,
    };

    // Start monitoring withdrawal status
    this.startMonitoring(result.withdrawalId, 'withdraw');

    return withdrawalInfo;
  }

  /**
   * Get deposit information and current status
   */
  async getDepositInfo(depositId: string): Promise<DepositInfo | null> {
    try {
      const info = await this.client.getBridgeStatus(depositId);
      return info as DepositInfo;
    } catch (error) {
      console.error('Failed to get deposit info:', error);
      return null;
    }
  }

  /**
   * Get withdrawal information and current status
   */
  async getWithdrawalInfo(withdrawalId: string): Promise<WithdrawalInfo | null> {
    try {
      const info = await this.client.getBridgeStatus(withdrawalId);
      return info as WithdrawalInfo;
    } catch (error) {
      console.error('Failed to get withdrawal info:', error);
      return null;
    }
  }

  /**
   * Wait for deposit to complete with timeout
   */
  async waitForDeposit(
    depositId: string,
    timeoutMs?: number
  ): Promise<DepositInfo> {
    return this.waitForCompletion(
      depositId,
      'deposit',
      timeoutMs || this.timeout
    );
  }

  /**
   * Wait for withdrawal to complete with timeout
   */
  async waitForWithdrawal(
    withdrawalId: string,
    timeoutMs?: number
  ): Promise<WithdrawalInfo> {
    return this.waitForCompletion(
      withdrawalId,
      'withdraw',
      timeoutMs || this.timeout
    );
  }

  /**
   * Get bridge statistics from the RPC endpoint
   */
  async getStats(): Promise<{
    totalDeposits: number;
    totalWithdrawals: number;
    totalVolume: number;
    activeRelayers: number;
  }> {
    try {
      // Fetch bridge stats from RPC
      const stats = await this.client.call<{
        totalDeposits: number;
        totalWithdrawals: number;
        totalDepositVolume: number;
        totalWithdrawalVolume: number;
        activeRelayers: number;
      }>('getBridgeStats', {});

      return {
        totalDeposits: stats.totalDeposits || 0,
        totalWithdrawals: stats.totalWithdrawals || 0,
        totalVolume: (stats.totalDepositVolume || 0) + (stats.totalWithdrawalVolume || 0),
        activeRelayers: stats.activeRelayers || 0,
      };
    } catch (error) {
      console.warn('Failed to fetch bridge stats from RPC:', error);
      // Return cached or default values
      return {
        totalDeposits: 0,
        totalWithdrawals: 0,
        totalVolume: 0,
        activeRelayers: 0,
      };
    }
  }

  /**
   * Estimate bridge fee by querying the RPC endpoint
   */
  async estimateFee(
    fromChain: string,
    toChain: string,
    amount: number
  ): Promise<number> {
    try {
      // Query dynamic fee from RPC
      const feeInfo = await this.client.call<{
        baseFee: number;
        percentageFee: number;
        totalFee: number;
        congestionFactor?: number;
      }>('estimateBridgeFee', {
        fromChain,
        toChain,
        amount,
      });

      return feeInfo.totalFee;
    } catch (error) {
      console.warn('Failed to estimate fee from RPC, using fallback:', error);
      // Fallback calculation
      const baseFee = 1000;
      const percentageFee = Math.floor(amount * 0.001); // 0.1%
      return baseFee + percentageFee;
    }
  }

  /**
   * Check if bridge is operational
   */
  async isOperational(): Promise<boolean> {
    try {
      const health = await this.client.getHealth();
      return health.status === 'ok';
    } catch {
      return false;
    }
  }

  /**
   * Start monitoring a bridge operation
   */
  private startMonitoring(id: string, type: 'deposit' | 'withdraw'): void {
    if (this.activePolls.has(id)) {
      return;
    }

    const poll = setInterval(async () => {
      try {
        const info =
          type === 'deposit'
            ? await this.getDepositInfo(id)
            : await this.getWithdrawalInfo(id);

        if (!info) {
          return;
        }

        const event: BridgeEvent = {
          type,
          id,
          status: info.status,
          data: info,
          timestamp: Date.now(),
        };

        this.emit('statusChange', event);
        this.emit(type, event);

        // Stop polling if completed or failed
        if (
          info.status === BridgeStatus.Completed ||
          info.status === BridgeStatus.Failed
        ) {
          this.stopMonitoring(id);
          this.emit('complete', event);
        }
      } catch (error) {
        console.error(`Error polling ${type} ${id}:`, error);
      }
    }, this.pollInterval);

    this.activePolls.set(id, poll);
  }

  /**
   * Stop monitoring a bridge operation
   */
  private stopMonitoring(id: string): void {
    const poll = this.activePolls.get(id);
    if (poll) {
      clearInterval(poll);
      this.activePolls.delete(id);
    }
  }

  /**
   * Wait for operation to complete
   */
  private async waitForCompletion(
    id: string,
    type: 'deposit' | 'withdraw',
    timeoutMs: number
  ): Promise<any> {
    return new Promise((resolve, reject) => {
      const timeoutHandle = setTimeout(() => {
        this.stopMonitoring(id);
        reject(new Error(`${type} timeout after ${timeoutMs}ms`));
      }, timeoutMs);

      const onComplete = (event: BridgeEvent) => {
        if (event.id === id) {
          clearTimeout(timeoutHandle);
          this.removeListener('complete', onComplete);

          if (event.status === BridgeStatus.Completed) {
            resolve(event.data);
          } else {
            reject(new Error(`${type} failed: ${event.status}`));
          }
        }
      };

      this.on('complete', onComplete);
      this.startMonitoring(id, type);
    });
  }

  /**
   * Cleanup - stop all active monitoring
   */
  destroy(): void {
    for (const [id] of this.activePolls) {
      this.stopMonitoring(id);
    }
    this.removeAllListeners();
  }
}

/**
 * Advanced bridge client with batch operations
 */
export class AdvancedBridgeClient extends BridgeClient {
  /**
   * Batch deposit multiple amounts
   */
  async batchDeposit(
    deposits: Array<{
      fromChain: string;
      amount: number;
      tokenAddress: string;
      recipient?: string;
    }>
  ): Promise<DepositInfo[]> {
    const results = await Promise.all(
      deposits.map((d) =>
        this.deposit(d.fromChain, d.amount, d.tokenAddress, d.recipient)
      )
    );
    return results;
  }

  /**
   * Batch withdraw multiple amounts
   */
  async batchWithdraw(
    withdrawals: Array<{
      toChain: string;
      amount: number;
      recipient: string;
      tokenAddress?: string;
    }>
  ): Promise<WithdrawalInfo[]> {
    const results = await Promise.all(
      withdrawals.map((w) =>
        this.withdraw(w.toChain, w.amount, w.recipient, w.tokenAddress)
      )
    );
    return results;
  }

  /**
   * Get transaction history for an address from the RPC endpoint
   */
  async getHistory(
    address: string,
    limit: number = 50
  ): Promise<Array<DepositInfo | WithdrawalInfo>> {
    try {
      const result = await this.client.call<{
        operations: Array<{
          id: string;
          type: 'deposit' | 'withdraw';
          status: string;
          amount: number;
          fromChain: string;
          toChain: string;
          token: string;
          sender: string;
          recipient: string;
          timestamp: number;
          txHash?: string;
          confirmations?: number;
        }>;
        total: number;
      }>('getBridgeHistory', { address, limit });

      return result.operations.map((op) => {
        const status = op.status as BridgeStatus;
        
        if (op.type === 'deposit') {
          return {
            depositId: op.id,
            fromChain: op.fromChain,
            toChain: op.toChain,
            amount: op.amount,
            token: op.token,
            sender: op.sender,
            recipient: op.recipient,
            status,
            timestamp: op.timestamp,
            confirmations: op.confirmations,
            txHash: op.txHash,
          } as DepositInfo;
        } else {
          return {
            withdrawalId: op.id,
            fromChain: op.fromChain,
            toChain: op.toChain,
            amount: op.amount,
            token: op.token,
            sender: op.sender,
            recipient: op.recipient,
            status,
            timestamp: op.timestamp,
            confirmations: op.confirmations,
            txHash: op.txHash,
          } as WithdrawalInfo;
        }
      });
    } catch (error) {
      console.warn('Failed to fetch bridge history:', error);
      return [];
    }
  }

  /**
   * Retry a failed operation
   */
  async retryOperation(id: string, type: 'deposit' | 'withdraw'): Promise<boolean> {
    try {
      const result = await this.client.call<{ success: boolean; signature?: string }>(
        'retryBridgeOperation',
        { operationId: id, type }
      );

      if (result.success) {
        // Resume monitoring
        this.startMonitoring(id, type);
        return true;
      }
      return false;
    } catch (error) {
      console.error('Failed to retry operation:', error);
      return false;
    }
  }

  /**
   * Cancel a pending deposit by calling the RPC endpoint
   */
  async cancelDeposit(depositId: string): Promise<boolean> {
    try {
      const result = await this.client.call<{ success: boolean; refundTxHash?: string }>(
        'cancelBridgeDeposit',
        { depositId }
      );

      if (result.success) {
        this.stopMonitoring(depositId);
        this.emit('cancelled', { id: depositId, refundTxHash: result.refundTxHash });
        return true;
      }
      return false;
    } catch (error) {
      console.error('Failed to cancel deposit:', error);
      return false;
    }
  }
}
