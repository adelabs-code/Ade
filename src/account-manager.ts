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
 * Token account manager with proper RPC integration
 */
export class TokenAccountManager {
  private balanceCache: Map<string, { balance: bigint; timestamp: number }> = new Map();
  private cacheExpiry: number = 5000; // 5 seconds

  constructor(private client: AdeSidechainClient) {}

  /**
   * Get token accounts by owner
   */
  async getTokenAccountsByOwner(
    owner: string,
    mint?: string
  ): Promise<TokenAccount[]> {
    try {
      const options = mint ? { mint } : {};
      const result = await this.client.getTokenAccountsByOwner(owner, options);

      if (!result || !result.value) {
        return [];
      }

      const accounts: TokenAccount[] = [];
      for (const item of result.value) {
        const parsed = await this.parseTokenAccount(item);
        if (parsed) {
          accounts.push(parsed);
        }
      }

      return accounts;
    } catch (error) {
      console.error('Error fetching token accounts:', error);
      return [];
    }
  }

  /**
   * Parse token account data from RPC response
   */
  private async parseTokenAccount(item: any): Promise<TokenAccount | null> {
    try {
      const pubkey = item.pubkey;
      const accountData = item.account;

      // Parse the token account data based on the SPL Token Account layout
      // Account data is base64 encoded
      let mint = '';
      let owner = '';
      let amount = BigInt(0);
      let decimals = 9;

      if (accountData.data) {
        const data = typeof accountData.data === 'string' 
          ? Buffer.from(accountData.data, 'base64')
          : accountData.data;

        // SPL Token Account layout:
        // - mint (32 bytes)
        // - owner (32 bytes)
        // - amount (8 bytes, little-endian u64)
        // - delegate option (36 bytes)
        // - state (1 byte)
        // - is_native option (12 bytes)
        // - delegated_amount (8 bytes)
        // - close_authority option (36 bytes)

        if (data.length >= 72) {
          // Extract mint (first 32 bytes)
          mint = this.encodeBase58(data.slice(0, 32));
          
          // Extract owner (bytes 32-64)
          owner = this.encodeBase58(data.slice(32, 64));
          
          // Extract amount (bytes 64-72, little-endian u64)
          amount = this.readUint64LE(data, 64);
        }

        // Try to get decimals from mint info
        try {
          const mintInfo = await this.client.call<{ decimals: number }>(
            'getTokenSupply',
            { mint }
          );
          decimals = mintInfo?.decimals ?? 9;
        } catch {
          decimals = 9; // Default to 9 decimals
        }
      }

      return {
        address: pubkey,
        mint,
        owner: owner || accountData.owner,
        amount,
        decimals,
      };
    } catch (error) {
      console.error('Error parsing token account:', error);
      return null;
    }
  }

  /**
   * Get token balance from RPC
   */
  async getTokenBalance(tokenAccount: string): Promise<bigint> {
    // Check cache first
    const cached = this.balanceCache.get(tokenAccount);
    if (cached && Date.now() - cached.timestamp < this.cacheExpiry) {
      return cached.balance;
    }

    try {
      // Try the getTokenAccountBalance RPC method first
      const result = await this.client.call<{
        value: {
          amount: string;
          decimals: number;
          uiAmount: number;
        };
      }>('getTokenAccountBalance', { pubkey: tokenAccount });

      if (result?.value?.amount) {
        const balance = BigInt(result.value.amount);
        this.balanceCache.set(tokenAccount, { balance, timestamp: Date.now() });
        return balance;
      }

      // Fallback: parse account data directly
      const accountInfo = await this.client.getAccountInfo(tokenAccount);
      if (accountInfo && accountInfo.data) {
        const data = typeof accountInfo.data === 'string'
          ? Buffer.from(accountInfo.data, 'base64')
          : accountInfo.data;

        if (data.length >= 72) {
          const balance = this.readUint64LE(data, 64);
          this.balanceCache.set(tokenAccount, { balance, timestamp: Date.now() });
          return balance;
        }
      }

      return BigInt(0);
    } catch (error) {
      console.error('Error fetching token balance:', error);
      
      // Return cached value if available, even if expired
      if (cached) {
        return cached.balance;
      }
      
      return BigInt(0);
    }
  }

  /**
   * Get all token balances for an owner
   */
  async getAllTokenBalances(owner: string): Promise<Map<string, bigint>> {
    const balances = new Map<string, bigint>();
    
    try {
      const accounts = await this.getTokenAccountsByOwner(owner);
      
      for (const account of accounts) {
        balances.set(account.mint, account.amount);
      }
    } catch (error) {
      console.error('Error fetching all token balances:', error);
    }

    return balances;
  }

  /**
   * Read a little-endian uint64 from a buffer
   */
  private readUint64LE(buffer: Buffer, offset: number): bigint {
    const low = buffer.readUInt32LE(offset);
    const high = buffer.readUInt32LE(offset + 4);
    return BigInt(low) + (BigInt(high) << 32n);
  }

  /**
   * Encode bytes to base58
   */
  private encodeBase58(bytes: Buffer): string {
    const ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz';
    
    if (bytes.length === 0) return '';
    
    // Count leading zeros
    let zeros = 0;
    for (let i = 0; i < bytes.length && bytes[i] === 0; i++) {
      zeros++;
    }
    
    // Convert to base58
    const size = Math.ceil(bytes.length * 138 / 100) + 1;
    const b58 = new Uint8Array(size);
    
    for (const byte of bytes) {
      let carry = byte;
      for (let j = size - 1; j >= 0; j--) {
        carry += 256 * b58[j];
        b58[j] = carry % 58;
        carry = Math.floor(carry / 58);
      }
    }
    
    // Skip leading zeros in base58 result
    let i = 0;
    while (i < size && b58[i] === 0) {
      i++;
    }
    
    // Build string
    let str = '1'.repeat(zeros);
    for (; i < size; i++) {
      str += ALPHABET[b58[i]];
    }
    
    return str;
  }

  /**
   * Invalidate cache for a specific account
   */
  invalidateCache(tokenAccount: string): void {
    this.balanceCache.delete(tokenAccount);
  }

  /**
   * Clear entire balance cache
   */
  clearCache(): void {
    this.balanceCache.clear();
  }
}


