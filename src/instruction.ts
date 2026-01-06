import { PublicKey } from '@solana/web3.js';
import { serialize } from 'borsh';

export interface AccountMeta {
  pubkey: Uint8Array;
  isSigner: boolean;
  isWritable: boolean;
}

export enum InstructionType {
  Transfer = 0,
  CreateAccount = 1,
  AIAgentDeploy = 2,
  AIAgentExecute = 3,
  AIAgentUpdate = 4,
  BridgeDeposit = 5,
  BridgeWithdraw = 6,
}

export class TransferInstruction {
  type = InstructionType.Transfer;
  
  constructor(
    public from: Uint8Array,
    public to: Uint8Array,
    public amount: bigint
  ) {}

  static create(
    from: PublicKey,
    to: PublicKey,
    amount: bigint,
    programId: PublicKey
  ): { accounts: AccountMeta[]; data: Uint8Array } {
    const accounts: AccountMeta[] = [
      { pubkey: from.toBytes(), isSigner: true, isWritable: true },
      { pubkey: to.toBytes(), isSigner: false, isWritable: true },
    ];

    const instruction = new TransferInstruction(
      from.toBytes(),
      to.toBytes(),
      amount
    );

    const data = Buffer.concat([
      Buffer.from([InstructionType.Transfer]),
      Buffer.from(instruction.from),
      Buffer.from(instruction.to),
      this.encodeU64(amount),
    ]);

    return { accounts, data };
  }

  private static encodeU64(value: bigint): Buffer {
    const buffer = Buffer.alloc(8);
    buffer.writeBigUInt64LE(value);
    return buffer;
  }
}

export class AIAgentDeployInstruction {
  type = InstructionType.AIAgentDeploy;

  constructor(
    public agentId: Uint8Array,
    public modelHash: Uint8Array,
    public config: Uint8Array
  ) {}

  static create(
    agentId: PublicKey,
    modelHash: string,
    config: any,
    owner: PublicKey,
    programId: PublicKey
  ): { accounts: AccountMeta[]; data: Uint8Array } {
    const accounts: AccountMeta[] = [
      { pubkey: owner.toBytes(), isSigner: true, isWritable: true },
      { pubkey: agentId.toBytes(), isSigner: false, isWritable: true },
    ];

    const modelHashBytes = Buffer.from(modelHash, 'utf-8');
    const configBytes = Buffer.from(JSON.stringify(config), 'utf-8');

    const data = Buffer.concat([
      Buffer.from([InstructionType.AIAgentDeploy]),
      Buffer.from([modelHashBytes.length]),
      modelHashBytes,
      Buffer.from([configBytes.length]),
      configBytes,
    ]);

    return { accounts, data };
  }
}

export class AIAgentExecuteInstruction {
  type = InstructionType.AIAgentExecute;

  constructor(
    public agentId: Uint8Array,
    public inputData: Uint8Array,
    public maxCompute: bigint
  ) {}

  static create(
    agentId: PublicKey,
    inputData: any,
    maxCompute: bigint,
    caller: PublicKey,
    programId: PublicKey
  ): { accounts: AccountMeta[]; data: Uint8Array } {
    const accounts: AccountMeta[] = [
      { pubkey: caller.toBytes(), isSigner: true, isWritable: true },
      { pubkey: agentId.toBytes(), isSigner: false, isWritable: true },
    ];

    const inputBytes = Buffer.from(JSON.stringify(inputData), 'utf-8');
    const computeBuffer = Buffer.alloc(8);
    computeBuffer.writeBigUInt64LE(maxCompute);

    const data = Buffer.concat([
      Buffer.from([InstructionType.AIAgentExecute]),
      Buffer.from(agentId.toBytes()),
      Buffer.from([inputBytes.length]),
      inputBytes,
      computeBuffer,
    ]);

    return { accounts, data };
  }
}

export class BridgeDepositInstruction {
  type = InstructionType.BridgeDeposit;

  constructor(
    public fromChain: string,
    public amount: bigint,
    public tokenAddress: Uint8Array
  ) {}

  static create(
    fromChain: string,
    amount: bigint,
    tokenAddress: PublicKey,
    depositor: PublicKey,
    programId: PublicKey
  ): { accounts: AccountMeta[]; data: Uint8Array } {
    const accounts: AccountMeta[] = [
      { pubkey: depositor.toBytes(), isSigner: true, isWritable: true },
      { pubkey: tokenAddress.toBytes(), isSigner: false, isWritable: true },
    ];

    const chainBytes = Buffer.from(fromChain, 'utf-8');
    const amountBuffer = Buffer.alloc(8);
    amountBuffer.writeBigUInt64LE(amount);

    const data = Buffer.concat([
      Buffer.from([InstructionType.BridgeDeposit]),
      Buffer.from([chainBytes.length]),
      chainBytes,
      amountBuffer,
      Buffer.from(tokenAddress.toBytes()),
    ]);

    return { accounts, data };
  }
}








