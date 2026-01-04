import { BridgeClient, DepositInfo, WithdrawalInfo, BridgeStatus, BridgeEvent } from './bridge';
import { AdeSidechainClient } from './client';

/**
 * Bridge transaction tracker
 */
export class BridgeTransactionTracker {
  private transactions: Map<string, BridgeTransaction> = new Map();

  addTransaction(tx: BridgeTransaction): void {
    this.transactions.set(tx.id, tx);
  }

  getTransaction(id: string): BridgeTransaction | undefined {
    return this.transactions.get(id);
  }

  updateStatus(id: string, status: BridgeStatus): void {
    const tx = this.transactions.get(id);
    if (tx) {
      tx.status = status;
      tx.lastUpdated = Date.now();
    }
  }

  getPendingTransactions(): BridgeTransaction[] {
    return Array.from(this.transactions.values()).filter(
      (tx) => tx.status === BridgeStatus.Pending || tx.status === BridgeStatus.Locked
    );
  }

  getCompletedTransactions(): BridgeTransaction[] {
    return Array.from(this.transactions.values()).filter(
      (tx) => tx.status === BridgeStatus.Completed
    );
  }

  getFailedTransactions(): BridgeTransaction[] {
    return Array.from(this.transactions.values()).filter(
      (tx) => tx.status === BridgeStatus.Failed
    );
  }

  clear(): void {
    this.transactions.clear();
  }
}

export interface BridgeTransaction {
  id: string;
  type: 'deposit' | 'withdraw';
  fromChain: string;
  toChain: string;
  amount: number;
  status: BridgeStatus;
  createdAt: number;
  lastUpdated: number;
  retryCount: number;
}

/**
 * Bridge fee estimator
 */
export class BridgeFeeEstimator {
  constructor(private client: AdeSidechainClient) {}

  async estimateDeposit(fromChain: string, amount: number): Promise<FeeEstimate> {
    const baseFee = await this.getBaseFee(fromChain, 'ade');
    const percentageFee = Math.floor(amount * 0.001);
    const networkFee = await this.getNetworkFee(fromChain);

    return {
      baseFee,
      percentageFee,
      networkFee,
      totalFee: baseFee + percentageFee + networkFee,
      estimatedTime: 60, // seconds
    };
  }

  async estimateWithdrawal(toChain: string, amount: number): Promise<FeeEstimate> {
    const baseFee = await this.getBaseFee('ade', toChain);
    const percentageFee = Math.floor(amount * 0.001);
    const networkFee = await this.getNetworkFee(toChain);

    return {
      baseFee,
      percentageFee,
      networkFee,
      totalFee: baseFee + percentageFee + networkFee,
      estimatedTime: 120, // seconds
    };
  }

  private async getBaseFee(fromChain: string, toChain: string): Promise<number> {
    // Would call bridge API for actual fees
    return 1000;
  }

  private async getNetworkFee(chain: string): Promise<number> {
    // Would query chain-specific fees
    return 500;
  }
}

export interface FeeEstimate {
  baseFee: number;
  percentageFee: number;
  networkFee: number;
  totalFee: number;
  estimatedTime: number;
}

/**
 * Bridge analytics
 */
export class BridgeAnalytics {
  private history: BridgeTransaction[] = [];

  addTransaction(tx: BridgeTransaction): void {
    this.history.push(tx);
  }

  getTotalVolume(timeWindowMs?: number): number {
    const cutoff = timeWindowMs ? Date.now() - timeWindowMs : 0;
    
    return this.history
      .filter((tx) => tx.createdAt >= cutoff)
      .reduce((sum, tx) => sum + tx.amount, 0);
  }

  getAverageCompletionTime(): number {
    const completed = this.history.filter((tx) => tx.status === BridgeStatus.Completed);
    
    if (completed.length === 0) {
      return 0;
    }

    const totalTime = completed.reduce(
      (sum, tx) => sum + (tx.lastUpdated - tx.createdAt),
      0
    );

    return totalTime / completed.length;
  }

  getSuccessRate(): number {
    if (this.history.length === 0) {
      return 0;
    }

    const completed = this.history.filter((tx) => tx.status === BridgeStatus.Completed).length;
    return (completed / this.history.length) * 100;
  }

  getStats(): BridgeAnalyticsStats {
    return {
      totalTransactions: this.history.length,
      totalVolume: this.getTotalVolume(),
      averageCompletionTimeMs: this.getAverageCompletionTime(),
      successRate: this.getSuccessRate(),
      pendingCount: this.history.filter((tx) => tx.status === BridgeStatus.Pending).length,
      completedCount: this.history.filter((tx) => tx.status === BridgeStatus.Completed).length,
      failedCount: this.history.filter((tx) => tx.status === BridgeStatus.Failed).length,
    };
  }
}

export interface BridgeAnalyticsStats {
  totalTransactions: number;
  totalVolume: number;
  averageCompletionTimeMs: number;
  successRate: number;
  pendingCount: number;
  completedCount: number;
  failedCount: number;
}

/**
 * Enhanced bridge client with all features
 */
export class EnhancedBridgeClient extends BridgeClient {
  private tracker: BridgeTransactionTracker;
  private feeEstimator: BridgeFeeEstimator;
  private analytics: BridgeAnalytics;

  constructor(client: AdeSidechainClient, options?: any) {
    super(client, options);
    this.tracker = new BridgeTransactionTracker();
    this.feeEstimator = new BridgeFeeEstimator(client);
    this.analytics = new BridgeAnalytics();

    // Listen to events for tracking
    this.on('deposit', (event: BridgeEvent) => {
      this.tracker.updateStatus(event.id, event.status);
    });

    this.on('withdraw', (event: BridgeEvent) => {
      this.tracker.updateStatus(event.id, event.status);
    });
  }

  async deposit(
    fromChain: string,
    amount: number,
    tokenAddress: string,
    recipient?: string
  ): Promise<DepositInfo> {
    const result = await super.deposit(fromChain, amount, tokenAddress, recipient);

    const tx: BridgeTransaction = {
      id: result.depositId,
      type: 'deposit',
      fromChain,
      toChain: 'ade',
      amount,
      status: BridgeStatus.Pending,
      createdAt: Date.now(),
      lastUpdated: Date.now(),
      retryCount: 0,
    };

    this.tracker.addTransaction(tx);
    this.analytics.addTransaction(tx);

    return result;
  }

  async withdraw(
    toChain: string,
    amount: number,
    recipient: string,
    tokenAddress?: string
  ): Promise<WithdrawalInfo> {
    const result = await super.withdraw(toChain, amount, recipient, tokenAddress);

    const tx: BridgeTransaction = {
      id: result.withdrawalId,
      type: 'withdraw',
      fromChain: 'ade',
      toChain,
      amount,
      status: BridgeStatus.Pending,
      createdAt: Date.now(),
      lastUpdated: Date.now(),
      retryCount: 0,
    };

    this.tracker.addTransaction(tx);
    this.analytics.addTransaction(tx);

    return result;
  }

  getTracker(): BridgeTransactionTracker {
    return this.tracker;
  }

  getFeeEstimator(): BridgeFeeEstimator {
    return this.feeEstimator;
  }

  getAnalytics(): BridgeAnalytics {
    return this.analytics;
  }
}


