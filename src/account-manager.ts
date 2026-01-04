import { AdeSidechainClient } from './client';
import { PublicKey } from '@solana/web3.js';

export interface Account {
  address: string;
  lamports: number;
  owner: string;
  executable: boolean;
  rentEpoch: number;
  data?: string | Buffer;
}

export interface TokenAccount {
  address: string;
  mint: string;
  owner: string;
  amount: bigint;
  decimals: number;
}

export interface AccountChangeCallback {
  (account: Account): void;
}

export class AccountManager {
  private accountCache: Map<string, { account: Account; timestamp: number }> = new Map();
  private subscriptions: Map<string, Set<AccountChangeCallback>> = new Map();
  private cacheExpiry: number = 5000; // 5 seconds

  constructor(private client: AdeSidechainClient) {}

  /**
   * Get account with caching
   */
  async getAccount(address: string, useCache: boolean = true): Promise<Account | null> {
    // Check cache
    if (useCache) {
      const cached = this.accountCache.get(address);
      if (cached && Date.now() - cached.timestamp < this.cacheExpiry) {
        return cached.account;
      }
    }

    // Fetch from RPC
    try {
      const result = await this.client.getAccountInfo(address);
      
      if (!result || !result.value) {
        return null;
      }

      const account: Account = {
        address,
        lamports: result.value.lamports,
        owner: result.value.owner,
        executable: result.value.executable,
        rentEpoch: result.value.rentEpoch,
        data: result.value.data,
      };

      // Update cache
      this.accountCache.set(address, {
        account,
        timestamp: Date.now(),
      });

      // Notify subscribers
      this.notifySubscribers(address, account);

      return account;
    } catch (error) {
      console.error('Error fetching account:', error);
      return null;
    }
  }

  /**
   * Get multiple accounts
   */
  async getMultipleAccounts(addresses: string[]): Promise<(Account | null)[]> {
    try {
      const result = await this.client.getMultipleAccounts(addresses);
      
      const accounts: (Account | null)[] = result.value.map((accountInfo: any, index: number) => {
        if (!accountInfo) {
          return null;
        }

        const account: Account = {
          address: addresses[index],
          lamports: accountInfo.lamports,
          owner: accountInfo.owner,
          executable: accountInfo.executable,
          rentEpoch: accountInfo.rentEpoch,
          data: accountInfo.data,
        };

        // Update cache
        this.accountCache.set(addresses[index], {
          account,
          timestamp: Date.now(),
        });

        return account;
      });

      return accounts;
    } catch (error) {
      console.error('Error fetching multiple accounts:', error);
      return addresses.map(() => null);
    }
  }

  /**
   * Get account balance
   */
  async getBalance(address: string): Promise<number> {
    const account = await this.getAccount(address);
    return account?.lamports || 0;
  }

  /**
   * Check if account exists
   */
  async accountExists(address: string): Promise<boolean> {
    const account = await this.getAccount(address);
    return account !== null;
  }

  /**
   * Subscribe to account changes
   */
  subscribe(address: string, callback: AccountChangeCallback): () => void {
    if (!this.subscriptions.has(address)) {
      this.subscriptions.set(address, new Set());
    }

    this.subscriptions.get(address)!.add(callback);

    // Return unsubscribe function
    return () => {
      const callbacks = this.subscriptions.get(address);
      if (callbacks) {
        callbacks.delete(callback);
        if (callbacks.size === 0) {
          this.subscriptions.delete(address);
        }
      }
    };
  }

  /**
   * Notify subscribers of account changes
   */
  private notifySubscribers(address: string, account: Account): void {
    const callbacks = this.subscriptions.get(address);
    if (callbacks) {
      callbacks.forEach((callback) => {
        try {
          callback(account);
        } catch (error) {
          console.error('Error in account callback:', error);
        }
      });
    }
  }

  /**
   * Invalidate cache for address
   */
  invalidateCache(address: string): void {
    this.accountCache.delete(address);
  }

  /**
   * Clear entire cache
   */
  clearCache(): void {
    this.accountCache.clear();
  }

  /**
   * Set cache expiry time
   */
  setCacheExpiry(ms: number): void {
    this.cacheExpiry = ms;
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): { size: number; hitRate: number } {
    return {
      size: this.accountCache.size,
      hitRate: 0, // Would need to track hits/misses
    };
  }

  /**
   * Prune expired cache entries
   */
  pruneCache(): number {
    const now = Date.now();
    let pruned = 0;

    for (const [address, cached] of this.accountCache.entries()) {
      if (now - cached.timestamp > this.cacheExpiry) {
        this.accountCache.delete(address);
        pruned++;
      }
    }

    return pruned;
  }
}

/**
 * Program account manager
 */
export class ProgramAccountManager {
  constructor(private client: AdeSidechainClient) {}

  /**
   * Get all accounts owned by a program
   */
  async getProgramAccounts(
    programId: string,
    filters?: Array<{
      memcmp?: { offset: number; bytes: string };
      dataSize?: number;
    }>
  ): Promise<Array<{ pubkey: string; account: Account }>> {
    const result = await this.client.getProgramAccounts(programId);
    
    return result.map((item: any) => ({
      pubkey: item.pubkey,
      account: {
        address: item.pubkey,
        lamports: item.account.lamports,
        owner: item.account.owner,
        executable: item.account.executable,
        rentEpoch: item.account.rentEpoch,
        data: item.account.data,
      },
    }));
  }

  /**
   * Get program accounts with filters
   */
  async getFilteredAccounts(
    programId: string,
    filter: (account: Account) => boolean
  ): Promise<Array<{ pubkey: string; account: Account }>> {
    const accounts = await this.getProgramAccounts(programId);
    return accounts.filter(({ account }) => filter(account));
  }

  /**
   * Get accounts by data size
   */
  async getAccountsByDataSize(programId: string, dataSize: number): Promise<any[]> {
    return this.getFilteredAccounts(programId, (account) => {
      if (typeof account.data === 'string') {
        return Buffer.from(account.data, 'base64').length === dataSize;
      }
      return false;
    });
  }
}

/**
 * Token account manager
 */
export class TokenAccountManager {
  constructor(private client: AdeSidechainClient) {}

  /**
   * Get token accounts by owner
   */
  async getTokenAccountsByOwner(
    owner: string,
    mint?: string
  ): Promise<TokenAccount[]> {
    const options = mint ? { mint } : {};
    const result = await this.client.getTokenAccountsByOwner(owner, options);

    if (!result.value) {
      return [];
    }

    return result.value.map((item: any) => this.parseTokenAccount(item));
  }

  /**
   * Parse token account data
   */
  private parseTokenAccount(item: any): TokenAccount {
    // Simplified parsing - in production would decode actual token account data
    return {
      address: item.pubkey,
      mint: 'TokenMintAddress',
      owner: item.account.owner,
      amount: BigInt(0),
      decimals: 9,
    };
  }

  /**
   * Get token balance
   */
  async getTokenBalance(tokenAccount: string): Promise<bigint> {
    const account = await this.client.getAccountInfo(tokenAccount);
    // Would parse token account data
    return BigInt(0);
  }
}


