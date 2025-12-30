import { AdeSidechainClient } from './client';

export enum BridgeStatus {
  Pending = 'pending',
  Locked = 'locked',
  Relayed = 'relayed',
  Completed = 'completed',
  Failed = 'failed',
}

export interface DepositInfo {
  depositId: string;
  fromChain: string;
  toChain: string;
  amount: number;
  token: string;
  sender: string;
  recipient: string;
  status: BridgeStatus;
  timestamp: number;
}

export interface WithdrawalInfo {
  withdrawalId: string;
  fromChain: string;
  toChain: string;
  amount: number;
  token: string;
  sender: string;
  recipient: string;
  status: BridgeStatus;
  timestamp: number;
}

export class BridgeClient {
  constructor(private client: AdeSidechainClient) {}

  async deposit(
    fromChain: string,
    amount: number,
    tokenAddress: string
  ): Promise<{ depositId: string; signature: string }> {
    return this.client.bridgeDeposit({
      fromChain,
      amount,
      tokenAddress,
    });
  }

  async withdraw(
    toChain: string,
    amount: number,
    recipient: string
  ): Promise<{ withdrawalId: string; signature: string }> {
    return this.client.bridgeWithdraw({
      toChain,
      amount,
      recipient,
    });
  }
}

