import { Keypair, PublicKey } from '@solana/web3.js';
import { Transaction, Message, MessageHeader, CompiledInstruction } from './transaction';
import * as crypto from 'crypto';

export interface TransactionOptions {
  feePayer?: PublicKey;
  recentBlockhash?: Uint8Array;
  nonceInfo?: {
    nonce: string;
    nonceInstruction: CompiledInstruction;
  };
}

export interface AddressLookupTable {
  key: PublicKey;
  addresses: PublicKey[];
}

export class AdvancedTransactionBuilder {
  private instructions: CompiledInstruction[] = [];
  private signers: Keypair[] = [];
  private feePayer?: Keypair;
  private recentBlockhash?: Uint8Array;
  private addressLookupTables: AddressLookupTable[] = [];
  private computeBudget?: number;
  private priorityFee?: number;

  constructor() {}

  /**
   * Add instruction to transaction
   */
  addInstruction(instruction: CompiledInstruction): this {
    this.instructions.push(instruction);
    return this;
  }

  /**
   * Add multiple instructions at once
   */
  addInstructions(instructions: CompiledInstruction[]): this {
    this.instructions.push(...instructions);
    return this;
  }

  /**
   * Add signer keypair
   */
  addSigner(signer: Keypair): this {
    if (!this.signers.find((s) => s.publicKey.equals(signer.publicKey))) {
      this.signers.push(signer);
    }
    return this;
  }

  /**
   * Set fee payer (defaults to first signer)
   */
  setFeePayer(feePayer: Keypair): this {
    this.feePayer = feePayer;
    return this;
  }

  /**
   * Set recent blockhash
   */
  setRecentBlockhash(blockhash: Uint8Array): this {
    this.recentBlockhash = blockhash;
    return this;
  }

  /**
   * Set compute budget
   */
  setComputeBudget(units: number): this {
    this.computeBudget = units;
    return this;
  }

  /**
   * Set priority fee
   */
  setPriorityFee(microLamports: number): this {
    this.priorityFee = microLamports;
    return this;
  }

  /**
   * Add address lookup table
   */
  addAddressLookupTable(table: AddressLookupTable): this {
    this.addressLookupTables.push(table);
    return this;
  }

  /**
   * Estimate transaction size
   */
  estimateSize(): number {
    // Approximate calculation
    let size = 1; // Signature count
    size += this.signers.length * 64; // Signatures
    size += 3; // Header
    size += this.getAccountKeys().length * 32; // Account keys
    size += 32; // Recent blockhash
    size += this.instructions.length * 50; // Instructions (approximate)
    return size;
  }

  /**
   * Check if transaction is within size limits
   */
  isWithinSizeLimit(): boolean {
    const MAX_TX_SIZE = 1232;
    return this.estimateSize() <= MAX_TX_SIZE;
  }

  /**
   * Get all account keys
   */
  private getAccountKeys(): Uint8Array[] {
    const feePayer = this.feePayer || this.signers[0];
    const keys = [feePayer.publicKey.toBytes()];

    // Add other signers
    for (const signer of this.signers) {
      if (!feePayer.publicKey.equals(signer.publicKey)) {
        keys.push(signer.publicKey.toBytes());
      }
    }

    return keys;
  }

  /**
   * Build the transaction
   */
  build(): Transaction {
    if (!this.recentBlockhash) {
      throw new Error('Recent blockhash not set');
    }

    if (this.signers.length === 0) {
      throw new Error('No signers provided');
    }

    // Add compute budget instruction if set
    if (this.computeBudget) {
      this.instructions.unshift(this.createComputeBudgetInstruction());
    }

    // Add priority fee instruction if set
    if (this.priorityFee) {
      this.instructions.unshift(this.createPriorityFeeInstruction());
    }

    const accountKeys = this.getAccountKeys();

    const message: Message = {
      header: {
        numRequiredSignatures: this.signers.length,
        numReadonlySignedAccounts: 0,
        numReadonlyUnsignedAccounts: 0,
      },
      accountKeys,
      recentBlockhash: this.recentBlockhash,
      instructions: this.instructions,
    };

    const transaction = new Transaction(message);
    transaction.sign(this.signers);

    return transaction;
  }

  /**
   * Build and sign transaction
   */
  async buildAndSign(): Promise<Transaction> {
    const tx = this.build();
    return tx;
  }

  /**
   * Create compute budget instruction
   */
  private createComputeBudgetInstruction(): CompiledInstruction {
    const data = Buffer.alloc(9);
    data.writeUInt8(0, 0); // SetComputeUnitLimit discriminator
    data.writeBigUInt64LE(BigInt(this.computeBudget!), 1);

    return {
      programIdIndex: 0,
      accounts: [],
      data,
    };
  }

  /**
   * Create priority fee instruction
   */
  private createPriorityFeeInstruction(): CompiledInstruction {
    const data = Buffer.alloc(9);
    data.writeUInt8(1, 0); // SetComputeUnitPrice discriminator
    data.writeBigUInt64LE(BigInt(this.priorityFee!), 1);

    return {
      programIdIndex: 0,
      accounts: [],
      data,
    };
  }

  /**
   * Clone the builder
   */
  clone(): AdvancedTransactionBuilder {
    const cloned = new AdvancedTransactionBuilder();
    cloned.instructions = [...this.instructions];
    cloned.signers = [...this.signers];
    cloned.feePayer = this.feePayer;
    cloned.recentBlockhash = this.recentBlockhash;
    cloned.computeBudget = this.computeBudget;
    cloned.priorityFee = this.priorityFee;
    return cloned;
  }

  /**
   * Clear all data
   */
  reset(): this {
    this.instructions = [];
    this.signers = [];
    this.feePayer = undefined;
    this.recentBlockhash = undefined;
    this.computeBudget = undefined;
    this.priorityFee = undefined;
    this.addressLookupTables = [];
    return this;
  }

  /**
   * Get instruction count
   */
  getInstructionCount(): number {
    return this.instructions.length;
  }

  /**
   * Get signer count
   */
  getSignerCount(): number {
    return this.signers.length;
  }
}

/**
 * Batch transaction builder
 */
export class BatchTransactionBuilder {
  private builders: AdvancedTransactionBuilder[] = [];

  addTransaction(builder: AdvancedTransactionBuilder): this {
    this.builders.push(builder);
    return this;
  }

  async buildAll(): Promise<Transaction[]> {
    return this.builders.map((builder) => builder.build());
  }

  getCount(): number {
    return this.builders.length;
  }

  clear(): this {
    this.builders = [];
    return this;
  }
}


