import { Keypair, PublicKey } from '@solana/web3.js';
import * as crypto from 'crypto';

export interface MessageHeader {
  numRequiredSignatures: number;
  numReadonlySignedAccounts: number;
  numReadonlyUnsignedAccounts: number;
}

export interface Message {
  header: MessageHeader;
  accountKeys: Uint8Array[];
  recentBlockhash: Uint8Array;
  instructions: CompiledInstruction[];
}

export interface CompiledInstruction {
  programIdIndex: number;
  accounts: number[];
  data: Uint8Array;
}

export class Transaction {
  public signatures: Uint8Array[] = [];
  public message: Message;

  constructor(message: Message) {
    this.message = message;
  }

  sign(signers: Keypair[]): void {
    const messageBytes = this.serializeMessage();
    
    this.signatures = signers.map(signer => {
      return signer.sign(messageBytes).signature;
    });
  }

  serialize(): Uint8Array {
    const messageBytes = this.serializeMessage();
    const signaturesLength = this.signatures.length;
    
    let totalLength = 1 + (signaturesLength * 64) + messageBytes.length;
    const buffer = new Uint8Array(totalLength);
    
    let offset = 0;
    buffer[offset++] = signaturesLength;
    
    for (const signature of this.signatures) {
      buffer.set(signature, offset);
      offset += 64;
    }
    
    buffer.set(messageBytes, offset);
    
    return buffer;
  }

  private serializeMessage(): Uint8Array {
    const buffers: Uint8Array[] = [];
    
    buffers.push(new Uint8Array([
      this.message.header.numRequiredSignatures,
      this.message.header.numReadonlySignedAccounts,
      this.message.header.numReadonlyUnsignedAccounts,
    ]));
    
    buffers.push(this.encodeLength(this.message.accountKeys.length));
    for (const key of this.message.accountKeys) {
      buffers.push(key);
    }
    
    buffers.push(this.message.recentBlockhash);
    
    buffers.push(this.encodeLength(this.message.instructions.length));
    for (const instruction of this.message.instructions) {
      buffers.push(new Uint8Array([instruction.programIdIndex]));
      buffers.push(this.encodeLength(instruction.accounts.length));
      buffers.push(new Uint8Array(instruction.accounts));
      buffers.push(this.encodeLength(instruction.data.length));
      buffers.push(instruction.data);
    }
    
    const totalLength = buffers.reduce((acc, buf) => acc + buf.length, 0);
    const result = new Uint8Array(totalLength);
    let offset = 0;
    
    for (const buffer of buffers) {
      result.set(buffer, offset);
      offset += buffer.length;
    }
    
    return result;
  }

  private encodeLength(length: number): Uint8Array {
    const bytes: number[] = [];
    let len = length;
    
    while (len > 0x7f) {
      bytes.push((len & 0x7f) | 0x80);
      len >>= 7;
    }
    bytes.push(len);
    
    return new Uint8Array(bytes);
  }

  hash(): Uint8Array {
    const serialized = this.serialize();
    return crypto.createHash('sha256').update(serialized).digest();
  }
}

export class TransactionBuilder {
  private instructions: CompiledInstruction[] = [];
  private signers: Keypair[] = [];
  private recentBlockhash?: Uint8Array;

  addInstruction(instruction: CompiledInstruction): this {
    this.instructions.push(instruction);
    return this;
  }

  addSigner(signer: Keypair): this {
    this.signers.push(signer);
    return this;
  }

  setRecentBlockhash(blockhash: Uint8Array): this {
    this.recentBlockhash = blockhash;
    return this;
  }

  build(): Transaction {
    if (!this.recentBlockhash) {
      throw new Error('Recent blockhash not set');
    }

    const accountKeys = this.signers.map(s => s.publicKey.toBytes());

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
}






