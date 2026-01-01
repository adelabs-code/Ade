/**
 * Enhanced Bridge Usage Examples
 */

import { createAdeClient } from '../api';
import { BridgeClient, BridgeStatus, BridgeEvent, AdvancedBridgeClient } from '../bridge';

async function basicBridgeUsage() {
  console.log('=== Basic Bridge Usage ===\n');

  const client = createAdeClient({
    rpcUrl: 'http://localhost:8899',
  });

  const bridge = new BridgeClient(client, {
    pollInterval: 5000,
    confirmationThreshold: 32,
    timeout: 300000,
  });

  // Deposit from Solana to Ade
  console.log('Initiating deposit from Solana...');
  const deposit = await bridge.deposit(
    'solana',
    1_000_000_000, // 1 SOL
    'So11111111111111111111111111111111111111112',
    'ade-recipient-address'
  );

  console.log('Deposit initiated:');
  console.log('  ID:', deposit.depositId);
  console.log('  Status:', deposit.status);
  console.log('  TX Hash:', deposit.txHash);

  // Listen for status changes
  bridge.on('statusChange', (event: BridgeEvent) => {
    console.log(`\nStatus update for ${event.id}:`);
    console.log('  Status:', event.status);
    console.log('  Confirmations:', event.data.confirmations);
  });

  bridge.on('complete', (event: BridgeEvent) => {
    console.log(`\n✅ Operation ${event.id} completed!`);
    console.log('  Final status:', event.status);
  });

  // Wait for completion
  try {
    const finalDeposit = await bridge.waitForDeposit(deposit.depositId);
    console.log('\nDeposit completed successfully!');
    console.log('  Amount:', finalDeposit.amount);
    console.log('  Recipient:', finalDeposit.recipient);
  } catch (error) {
    console.error('Deposit failed:', error);
  }
}

async function withdrawalExample() {
  console.log('\n=== Withdrawal Example ===\n');

  const client = createAdeClient({ rpcUrl: 'http://localhost:8899' });
  const bridge = new BridgeClient(client);

  // Withdraw from Ade to Solana
  const withdrawal = await bridge.withdraw(
    'solana',
    500_000_000, // 0.5 SOL
    'solana-recipient-address',
    'token-address'
  );

  console.log('Withdrawal initiated:', withdrawal.withdrawalId);

  // Poll for status updates
  const checkStatus = setInterval(async () => {
    const info = await bridge.getWithdrawalInfo(withdrawal.withdrawalId);
    if (!info) return;

    console.log(`Status: ${info.status}, Confirmations: ${info.confirmations}`);

    if (info.status === BridgeStatus.Completed) {
      console.log('✅ Withdrawal completed!');
      clearInterval(checkStatus);
      bridge.destroy();
    } else if (info.status === BridgeStatus.Failed) {
      console.log('❌ Withdrawal failed!');
      clearInterval(checkStatus);
      bridge.destroy();
    }
  }, 5000);
}

async function advancedFeatures() {
  console.log('\n=== Advanced Bridge Features ===\n');

  const client = createAdeClient({ rpcUrl: 'http://localhost:8899' });
  const bridge = new AdvancedBridgeClient(client);

  // 1. Fee estimation
  console.log('Estimating fees...');
  const fee = await bridge.estimateFee('solana', 'ade', 1_000_000_000);
  console.log(`Estimated fee: ${fee} lamports`);

  // 2. Check if bridge is operational
  const isOperational = await bridge.isOperational();
  console.log(`Bridge operational: ${isOperational}`);

  // 3. Batch operations
  console.log('\nExecuting batch deposits...');
  const deposits = await bridge.batchDeposit([
    {
      fromChain: 'solana',
      amount: 1_000_000_000,
      tokenAddress: 'token1',
      recipient: 'recipient1',
    },
    {
      fromChain: 'solana',
      amount: 2_000_000_000,
      tokenAddress: 'token1',
      recipient: 'recipient2',
    },
  ]);

  console.log(`Initiated ${deposits.length} deposits`);

  // 4. Get transaction history
  const history = await bridge.getHistory('my-address', 10);
  console.log(`\nTransaction history: ${history.length} operations`);

  // 5. Retry failed operation
  const failedOpId = 'some-failed-deposit-id';
  try {
    await bridge.retryOperation(failedOpId, 'deposit');
    console.log('Retry initiated');
  } catch (error) {
    console.error('Retry failed:', error);
  }

  // 6. Get bridge statistics
  const stats = await bridge.getStats();
  console.log('\nBridge Statistics:');
  console.log('  Total deposits:', stats.totalDeposits);
  console.log('  Total withdrawals:', stats.totalWithdrawals);
  console.log('  Total volume:', stats.totalVolume);
  console.log('  Active relayers:', stats.activeRelayers);
}

async function eventDrivenExample() {
  console.log('\n=== Event-Driven Bridge ===\n');

  const client = createAdeClient({ rpcUrl: 'http://localhost:8899' });
  const bridge = new BridgeClient(client);

  // Set up event listeners
  bridge.on('deposit', (event: BridgeEvent) => {
    console.log('📥 Deposit event:', event.id, '-', event.status);
  });

  bridge.on('withdraw', (event: BridgeEvent) => {
    console.log('📤 Withdrawal event:', event.id, '-', event.status);
  });

  bridge.on('statusChange', (event: BridgeEvent) => {
    console.log('🔄 Status changed:', event.id, 'to', event.status);
    
    // Custom logic based on status
    switch (event.status) {
      case BridgeStatus.Locked:
        console.log('  Funds locked, waiting for relay...');
        break;
      case BridgeStatus.Relayed:
        console.log('  Relayed to target chain, finalizing...');
        break;
      case BridgeStatus.Completed:
        console.log('  ✅ Operation completed!');
        break;
      case BridgeStatus.Failed:
        console.log('  ❌ Operation failed!');
        break;
    }
  });

  // Initiate operations
  const deposit = await bridge.deposit(
    'solana',
    1_000_000_000,
    'token-address'
  );

  // Wait for events
  await new Promise((resolve) => {
    bridge.once('complete', () => {
      console.log('\nAll operations completed!');
      bridge.destroy();
      resolve(null);
    });
  });
}

async function errorHandlingExample() {
  console.log('\n=== Error Handling ===\n');

  const client = createAdeClient({ rpcUrl: 'http://localhost:8899' });
  const bridge = new BridgeClient(client, {
    timeout: 60000, // 1 minute timeout
  });

  try {
    const deposit = await bridge.deposit(
      'solana',
      1_000_000_000,
      'invalid-token-address'
    );

    // Wait with timeout
    const result = await Promise.race([
      bridge.waitForDeposit(deposit.depositId),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Custom timeout')), 30000)
      ),
    ]);

    console.log('Deposit successful:', result);
  } catch (error) {
    console.error('Error occurred:', error);

    // Cancel pending operation
    // await bridge.cancelDeposit(deposit.depositId);
  } finally {
    bridge.destroy();
  }
}

// Main execution
async function main() {
  try {
    await basicBridgeUsage();
    // await withdrawalExample();
    // await advancedFeatures();
    // await eventDrivenExample();
    // await errorHandlingExample();
  } catch (error) {
    console.error('Error in bridge examples:', error);
  }
}

if (require.main === module) {
  main().catch(console.error);
}

export {
  basicBridgeUsage,
  withdrawalExample,
  advancedFeatures,
  eventDrivenExample,
  errorHandlingExample,
};


