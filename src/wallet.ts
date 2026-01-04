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
 * Implements BIP32/BIP44 compliant key derivation for Solana/Ade
 */
export class HDWallet {
  private seed: Buffer;
  private masterKey: Buffer;
  private chainCode: Buffer;

  constructor(seed: Buffer) {
    this.seed = seed;
    // Derive master key and chain code using HMAC-SHA512
    const hmac = crypto.createHmac('sha512', 'ed25519 seed');
    hmac.update(seed);
    const I = hmac.digest();
    this.masterKey = I.subarray(0, 32);
    this.chainCode = I.subarray(32, 64);
  }

  /**
   * Create from mnemonic using BIP39 standard
   */
  static fromMnemonic(mnemonic: string, passphrase: string = ''): HDWallet {
    // BIP39: Use PBKDF2 with 2048 iterations and mnemonic + "mnemonic" + passphrase as salt
    const salt = 'mnemonic' + passphrase;
    const seed = crypto.pbkdf2Sync(mnemonic, salt, 2048, 64, 'sha512');
    return new HDWallet(seed);
  }

  /**
   * Derive keypair at path using SLIP-0010/BIP32-Ed25519
   * Path format: m/44'/501'/{account}'/0'/{index}'
   * 44' = BIP44 purpose
   * 501' = Solana coin type
   */
  derivePath(path: string): Keypair {
    // Parse path components
    const components = path.replace('m/', '').split('/');
    
    let key = this.masterKey;
    let chainCode = this.chainCode;
    
    for (const component of components) {
      if (!component) continue;
      
      // Check if hardened derivation (ends with ')
      const hardened = component.endsWith("'");
      let index = parseInt(hardened ? component.slice(0, -1) : component, 10);
      
      if (hardened) {
        // Add hardened flag (0x80000000)
        index += 0x80000000;
      }
      
      // Derive child key using HMAC-SHA512
      const data = Buffer.alloc(37);
      data[0] = 0x00; // Ed25519 always uses hardened derivation
      key.copy(data, 1);
      data.writeUInt32BE(index, 33);
      
      const hmac = crypto.createHmac('sha512', chainCode);
      hmac.update(data);
      const I = hmac.digest();
      
      key = I.subarray(0, 32);
      chainCode = I.subarray(32, 64);
    }
    
    // Create Ed25519 keypair from derived key
    // The derived key is the seed for Keypair.fromSeed
    return Keypair.fromSeed(new Uint8Array(key));
  }

  /**
   * Get wallet at index using Solana's derivation path
   * Standard path: m/44'/501'/{index}'/0'
   */
  getWallet(index: number): Wallet {
    const path = `m/44'/501'/${index}'/0'`;
    const keypair = this.derivePath(path);
    return new Wallet(keypair);
  }

  /**
   * Get wallet with custom account and change indexes
   * Path: m/44'/501'/{account}'/{change}'
   */
  getWalletWithPath(account: number, change: number = 0): Wallet {
    const path = `m/44'/501'/${account}'/${change}'`;
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

  /**
   * Get the master public key (for watch-only wallets)
   */
  getMasterPublicKey(): PublicKey {
    const keypair = Keypair.fromSeed(new Uint8Array(this.masterKey));
    return keypair.publicKey;
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


