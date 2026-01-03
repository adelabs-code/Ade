import { Keypair, PublicKey } from '@solana/web3.js';
import * as crypto from 'crypto';
import * as fs from 'fs';

/**
 * Wallet utilities
 */
export class Wallet {
  constructor(public keypair: Keypair) {}

  /**
   * Create new wallet
   */
  static generate(): Wallet {
    const keypair = Keypair.generate();
    return new Wallet(keypair);
  }

  /**
   * Create wallet from secret key
   */
  static fromSecretKey(secretKey: Uint8Array): Wallet {
    const keypair = Keypair.fromSecretKey(secretKey);
    return new Wallet(keypair);
  }

  /**
   * Load wallet from file
   */
  static fromFile(path: string): Wallet {
    const data = fs.readFileSync(path);
    const secretKey = new Uint8Array(JSON.parse(data.toString()));
    return Wallet.fromSecretKey(secretKey);
  }

  /**
   * Save wallet to file
   */
  saveToFile(path: string): void {
    const secretKey = Array.from(this.keypair.secretKey);
    fs.writeFileSync(path, JSON.stringify(secretKey));
  }

  /**
   * Get public key
   */
  get publicKey(): PublicKey {
    return this.keypair.publicKey;
  }

  /**
   * Get address as string
   */
  get address(): string {
    return this.keypair.publicKey.toBase58();
  }

  /**
   * Sign message
   */
  sign(message: Uint8Array): Uint8Array {
    return this.keypair.sign(message).signature;
  }

  /**
   * Export as JSON
   */
  toJSON(): any {
    return {
      publicKey: this.publicKey.toBase58(),
      secretKey: Array.from(this.keypair.secretKey),
    };
  }
}

/**
 * HD Wallet (Hierarchical Deterministic)
 */
export class HDWallet {
  private seed: Buffer;

  constructor(seed: Buffer) {
    this.seed = seed;
  }

  /**
   * Create from mnemonic
   */
  static fromMnemonic(mnemonic: string, passphrase: string = ''): HDWallet {
    // In production, use bip39 library
    const seed = crypto.pbkdf2Sync(mnemonic, passphrase, 2048, 64, 'sha512');
    return new HDWallet(seed);
  }

  /**
   * Derive keypair at path
   */
  derivePath(path: string): Keypair {
    // Simplified derivation - in production use proper BIP32/44
    const pathHash = crypto.createHash('sha256').update(path).update(this.seed).digest();
    const secretKey = new Uint8Array(pathHash);
    
    // Pad to 64 bytes for ed25519
    const fullKey = new Uint8Array(64);
    fullKey.set(secretKey);
    
    return Keypair.fromSecretKey(fullKey);
  }

  /**
   * Get wallet at index
   */
  getWallet(index: number): Wallet {
    const path = `m/44'/501'/${index}'/0'`;
    const keypair = this.derivePath(path);
    return new Wallet(keypair);
  }

  /**
   * Get multiple wallets
   */
  getWallets(count: number, startIndex: number = 0): Wallet[] {
    const wallets: Wallet[] = [];
    
    for (let i = 0; i < count; i++) {
      wallets.push(this.getWallet(startIndex + i));
    }

    return wallets;
  }
}

/**
 * Multi-signature wallet
 */
export class MultiSigWallet {
  constructor(
    public signers: PublicKey[],
    public threshold: number
  ) {
    if (threshold > signers.length) {
      throw new Error('Threshold cannot exceed number of signers');
    }
  }

  /**
   * Check if signatures meet threshold
   */
  hasThreshold(signatureCount: number): boolean {
    return signatureCount >= this.threshold;
  }

  /**
   * Get required signatures
   */
  getRequiredSignatures(): number {
    return this.threshold;
  }

  /**
   * Check if pubkey is a signer
   */
  isSigner(pubkey: PublicKey): boolean {
    return this.signers.some((s) => s.equals(pubkey));
  }
}

/**
 * Keypair utilities
 */
export class KeypairUtils {
  /**
   * Generate random keypair
   */
  static generate(): Keypair {
    return Keypair.generate();
  }

  /**
   * Generate from seed
   */
  static fromSeed(seed: Uint8Array): Keypair {
    if (seed.length !== 32) {
      throw new Error('Seed must be 32 bytes');
    }
    return Keypair.fromSeed(seed);
  }

  /**
   * Create from secret key bytes
   */
  static fromSecretKey(secretKey: Uint8Array): Keypair {
    return Keypair.fromSecretKey(secretKey);
  }

  /**
   * Verify signature
   */
  static verify(
    message: Uint8Array,
    signature: Uint8Array,
    publicKey: PublicKey
  ): boolean {
    try {
      return publicKey.verify(message, signature);
    } catch {
      return false;
    }
  }

  /**
   * Generate deterministic keypair from string
   */
  static fromString(str: string): Keypair {
    const hash = crypto.createHash('sha256').update(str).digest();
    return this.fromSeed(new Uint8Array(hash));
  }

  /**
   * Batch generate keypairs
   */
  static generateBatch(count: number): Keypair[] {
    return Array.from({ length: count }, () => Keypair.generate());
  }
}

/**
 * Address utilities
 */
export class AddressUtils {
  /**
   * Validate address format
   */
  static isValidAddress(address: string): boolean {
    try {
      new PublicKey(address);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Convert to bytes
   */
  static toBytes(address: string): Uint8Array {
    return new PublicKey(address).toBytes();
  }

  /**
   * Convert from bytes
   */
  static fromBytes(bytes: Uint8Array): string {
    return new PublicKey(bytes).toBase58();
  }

  /**
   * Shorten address for display
   */
  static shorten(address: string, chars: number = 4): string {
    if (address.length <= chars * 2) {
      return address;
    }
    return `${address.slice(0, chars)}...${address.slice(-chars)}`;
  }

  /**
   * Generate PDA (Program Derived Address)
   */
  static async findProgramAddress(
    seeds: Array<Buffer | Uint8Array>,
    programId: PublicKey
  ): Promise<[PublicKey, number]> {
    return PublicKey.findProgramAddress(seeds, programId);
  }
}

