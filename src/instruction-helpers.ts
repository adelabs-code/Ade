import { PublicKey } from '@solana/web3.js';
import { AccountMeta } from './instruction';

/**
 * Instruction helper utilities
 */
export class InstructionHelpers {
  /**
   * Create transfer instruction data
   */
  static createTransferData(amount: bigint): Uint8Array {
    const buffer = Buffer.alloc(17);
    buffer.writeUInt8(0, 0); // Transfer discriminator
    buffer.writeBigUInt64LE(amount, 1);
    return new Uint8Array(buffer);
  }

  /**
   * Create account instruction data
   */
  static createAccountData(space: bigint, lamports: bigint): Uint8Array {
    const buffer = Buffer.alloc(25);
    buffer.writeUInt8(1, 0); // CreateAccount discriminator
    buffer.writeBigUInt64LE(space, 1);
    buffer.writeBigUInt64LE(lamports, 9);
    return new Uint8Array(buffer);
  }

  /**
   * Parse instruction discriminator
   */
  static parseDiscriminator(data: Uint8Array): number {
    return data[0];
  }

  /**
   * Validate account metas
   */
  static validateAccounts(accounts: AccountMeta[]): boolean {
    return accounts.every((acc) => acc.pubkey.length === 32);
  }

  /**
   * Find signer accounts
   */
  static getSigners(accounts: AccountMeta[]): AccountMeta[] {
    return accounts.filter((acc) => acc.isSigner);
  }

  /**
   * Find writable accounts
   */
  static getWritableAccounts(accounts: AccountMeta[]): AccountMeta[] {
    return accounts.filter((acc) => acc.isWritable);
  }

  /**
   * Deduplicate account metas
   */
  static deduplicateAccounts(accounts: AccountMeta[]): AccountMeta[] {
    const seen = new Set<string>();
    const result: AccountMeta[] = [];

    for (const account of accounts) {
      const key = Buffer.from(account.pubkey).toString('base64');
      if (!seen.has(key)) {
        seen.add(key);
        result.push(account);
      }
    }

    return result;
  }

  /**
   * Merge account privileges (signer, writable)
   */
  static mergeAccountMetas(accounts: AccountMeta[]): AccountMeta[] {
    const accountMap = new Map<string, AccountMeta>();

    for (const account of accounts) {
      const key = Buffer.from(account.pubkey).toString('base64');
      const existing = accountMap.get(key);

      if (existing) {
        // Merge privileges
        existing.isSigner = existing.isSigner || account.isSigner;
        existing.isWritable = existing.isWritable || account.isWritable;
      } else {
        accountMap.set(key, { ...account });
      }
    }

    return Array.from(accountMap.values());
  }
}

/**
 * System program helper
 */
export class SystemProgram {
  static readonly PROGRAM_ID = new PublicKey('11111111111111111111111111111111');

  static transfer(params: {
    fromPubkey: PublicKey;
    toPubkey: PublicKey;
    lamports: bigint;
  }): { accounts: AccountMeta[]; data: Uint8Array } {
    return {
      accounts: [
        { pubkey: params.fromPubkey.toBytes(), isSigner: true, isWritable: true },
        { pubkey: params.toPubkey.toBytes(), isSigner: false, isWritable: true },
      ],
      data: InstructionHelpers.createTransferData(params.lamports),
    };
  }

  static createAccount(params: {
    fromPubkey: PublicKey;
    newAccountPubkey: PublicKey;
    lamports: bigint;
    space: bigint;
    programId: PublicKey;
  }): { accounts: AccountMeta[]; data: Uint8Array } {
    return {
      accounts: [
        { pubkey: params.fromPubkey.toBytes(), isSigner: true, isWritable: true },
        { pubkey: params.newAccountPubkey.toBytes(), isSigner: true, isWritable: true },
      ],
      data: InstructionHelpers.createAccountData(params.space, params.lamports),
    };
  }
}

/**
 * Compute budget program helper
 */
export class ComputeBudgetProgram {
  static setComputeUnitLimit(units: number): { accounts: AccountMeta[]; data: Uint8Array } {
    const data = Buffer.alloc(9);
    data.writeUInt8(0, 0);
    data.writeBigUInt64LE(BigInt(units), 1);

    return {
      accounts: [],
      data: new Uint8Array(data),
    };
  }

  static setComputeUnitPrice(microLamports: number): { accounts: AccountMeta[]; data: Uint8Array } {
    const data = Buffer.alloc(9);
    data.writeUInt8(1, 0);
    data.writeBigUInt64LE(BigInt(microLamports), 1);

    return {
      accounts: [],
      data: new Uint8Array(data),
    };
  }
}

/**
 * Instruction validator
 */
export class InstructionValidator {
  static readonly MAX_INSTRUCTION_DATA_SIZE = 10240; // 10KB
  static readonly MAX_ACCOUNTS = 256;

  static validate(instruction: {
    accounts: AccountMeta[];
    data: Uint8Array;
  }): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Validate account count
    if (instruction.accounts.length > this.MAX_ACCOUNTS) {
      errors.push(`Too many accounts: ${instruction.accounts.length} > ${this.MAX_ACCOUNTS}`);
    }

    // Validate data size
    if (instruction.data.length > this.MAX_INSTRUCTION_DATA_SIZE) {
      errors.push(`Data too large: ${instruction.data.length} > ${this.MAX_INSTRUCTION_DATA_SIZE}`);
    }

    // Validate account pubkeys
    for (let i = 0; i < instruction.accounts.length; i++) {
      if (instruction.accounts[i].pubkey.length !== 32) {
        errors.push(`Invalid pubkey length at index ${i}`);
      }
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  }
}

/**
 * Fee calculator
 */
export class FeeCalculator {
  static readonly LAMPORTS_PER_SIGNATURE = 5000;

  static calculateFee(numSignatures: number): number {
    return numSignatures * this.LAMPORTS_PER_SIGNATURE;
  }

  static estimateTransactionFee(params: {
    numSignatures: number;
    computeUnits?: number;
    priorityFee?: number;
  }): number {
    let fee = this.calculateFee(params.numSignatures);

    if (params.computeUnits) {
      // Estimate compute fee
      fee += Math.ceil(params.computeUnits / 100);
    }

    if (params.priorityFee) {
      fee += params.priorityFee;
    }

    return fee;
  }
}


