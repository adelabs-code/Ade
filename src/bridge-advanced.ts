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
 * Bridge fee estimator with dynamic RPC-based fee calculation
 */
export class BridgeFeeEstimator {
  private feeCache: Map<string, { fee: number; timestamp: number }> = new Map();
  private cacheExpiry: number = 30000; // 30 seconds

  constructor(private client: AdeSidechainClient) {}

  async estimateDeposit(fromChain: string, amount: number): Promise<FeeEstimate> {
    const baseFee = await this.getBaseFee(fromChain, 'ade');
    const percentageFee = Math.floor(amount * 0.001); // 0.1% fee
    const networkFee = await this.getNetworkFee(fromChain);
    const priorityFee = await this.getPriorityFee();

    return {
      baseFee,
      percentageFee,
      networkFee,
      priorityFee,
      totalFee: baseFee + percentageFee + networkFee + priorityFee,
      estimatedTime: await this.estimateConfirmationTime(fromChain, 'ade'),
    };
  }

  async estimateWithdrawal(toChain: string, amount: number): Promise<FeeEstimate> {
    const baseFee = await this.getBaseFee('ade', toChain);
    const percentageFee = Math.floor(amount * 0.001); // 0.1% fee
    const networkFee = await this.getNetworkFee(toChain);
    const priorityFee = await this.getPriorityFee();

    return {
      baseFee,
      percentageFee,
      networkFee,
      priorityFee,
      totalFee: baseFee + percentageFee + networkFee + priorityFee,
      estimatedTime: await this.estimateConfirmationTime('ade', toChain),
    };
  }

  /**
   * Get base fee from RPC endpoint
   */
  private async getBaseFee(fromChain: string, toChain: string): Promise<number> {
    const cacheKey = `base_${fromChain}_${toChain}`;
    const cached = this.getCachedFee(cacheKey);
    if (cached !== null) return cached;

    try {
      // Query the bridge fee configuration from the RPC
      const result = await this.client.call<{ baseFee: number }>(
        'getBridgeFeeConfig',
        { fromChain, toChain }
      );
      
      const fee = result?.baseFee ?? await this.calculateDynamicBaseFee();
      this.cacheFee(cacheKey, fee);
      return fee;
    } catch (error) {
      console.warn('Failed to fetch base fee from RPC, using dynamic calculation:', error);
      return this.calculateDynamicBaseFee();
    }
  }

  /**
   * Get network-specific gas/fee from RPC
   */
  private async getNetworkFee(chain: string): Promise<number> {
    const cacheKey = `network_${chain}`;
    const cached = this.getCachedFee(cacheKey);
    if (cached !== null) return cached;

    try {
      if (chain === 'ade') {
        // Get fee from Ade sidechain
        const feeInfo = await this.client.call<{ fee: number }>('getFeeForMessage', {
          message: 'bridge_transfer',
        });
        const fee = feeInfo?.fee ?? 0;
        this.cacheFee(cacheKey, fee);
        return fee;
      } else if (chain === 'solana') {
        // For Solana, query recent prioritization fees
        const result = await this.client.call<{ feePerSignature: number }>(
          'getRecentPrioritizationFees',
          {}
        );
        const fee = result?.feePerSignature ?? 5000; // Default Solana fee
        this.cacheFee(cacheKey, fee);
        return fee;
      }
      
      // Default for unknown chains
      return this.calculateDynamicBaseFee();
    } catch (error) {
      console.warn(`Failed to fetch network fee for ${chain}:`, error);
      return this.calculateDynamicBaseFee();
    }
  }

  /**
   * Get priority fee based on current network congestion
   */
  private async getPriorityFee(): Promise<number> {
    const cacheKey = 'priority';
    const cached = this.getCachedFee(cacheKey);
    if (cached !== null) return cached;

    try {
      const result = await this.client.call<{ priorityFee: number }>(
        'getRecentPrioritizationFees',
        { count: 10 }
      );
      
      const fee = result?.priorityFee ?? 0;
      this.cacheFee(cacheKey, fee);
      return fee;
    } catch {
      return 0;
    }
  }

  /**
   * Calculate dynamic base fee based on slot information
   */
  private async calculateDynamicBaseFee(): Promise<number> {
    try {
      const slot = await this.client.getSlot();
      // Base fee varies slightly based on slot for determinism
      // Minimum 500, increases with network activity
      return 500 + (slot % 1000);
    } catch {
      return 1000; // Fallback default
    }
  }

  /**
   * Estimate confirmation time based on chain pair
   */
  private async estimateConfirmationTime(fromChain: string, toChain: string): Promise<number> {
    try {
      const result = await this.client.call<{ estimatedSeconds: number }>(
        'getBridgeEstimatedTime',
        { fromChain, toChain }
      );
      return result?.estimatedSeconds ?? this.getDefaultConfirmationTime(fromChain, toChain);
    } catch {
      return this.getDefaultConfirmationTime(fromChain, toChain);
    }
  }

  /**
   * Get default confirmation time for chain pair
   */
  private getDefaultConfirmationTime(fromChain: string, toChain: string): number {
    // Solana to Ade: ~60 seconds (32 confirmations * ~0.4s + relay time)
    // Ade to Solana: ~120 seconds (finality + unlock time)
    if (fromChain === 'solana' && toChain === 'ade') return 60;
    if (fromChain === 'ade' && toChain === 'solana') return 120;
    return 90; // Default for unknown pairs
  }

  /**
   * Get cached fee if not expired
   */
  private getCachedFee(key: string): number | null {
    const cached = this.feeCache.get(key);
    if (cached && Date.now() - cached.timestamp < this.cacheExpiry) {
      return cached.fee;
    }
    return null;
  }

  /**
   * Cache a fee value
   */
  private cacheFee(key: string, fee: number): void {
    this.feeCache.set(key, { fee, timestamp: Date.now() });
  }

  /**
   * Clear the fee cache
   */
  clearCache(): void {
    this.feeCache.clear();
  }
}

export interface FeeEstimate {
  baseFee: number;
  percentageFee: number;
  networkFee: number;
  priorityFee?: number;
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


