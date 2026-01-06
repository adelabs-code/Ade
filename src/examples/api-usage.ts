/**
 * Ade Sidechain API Usage Examples
 */

import { createAdeClient, AdeError, ErrorCode } from '../api';
import type {
  AIAgentConfig,
  BridgeStatus,
  Commitment,
} from '../types';

async function main() {
  // Initialize client
  const client = createAdeClient({
    rpcUrl: 'http://localhost:8899',
    timeout: 30000,
    retryAttempts: 3,
    commitment: 'confirmed',
  });

  try {
    // 1. Get chain state
    console.log('=== Chain State ===');
    const slot = await client.getSlot();
    console.log('Current slot:', slot);

    const blockHeight = await client.getBlockHeight();
    console.log('Block height:', blockHeight);

    const { blockhash } = await client.getLatestBlockhash();
    console.log('Latest blockhash:', blockhash);

    // 2. Account operations
    console.log('\n=== Account Operations ===');
    const address = 'your-address-here';
    
    const balance = await client.getBalance(address);
    console.log('Balance:', balance.value, 'lamports');

    const accountInfo = await client.getAccountInfo(address);
    console.log('Account owner:', accountInfo.owner);
    console.log('Is executable:', accountInfo.executable);

    // 3. AI Agent deployment
    console.log('\n=== AI Agent Operations ===');
    
    const agentConfig: AIAgentConfig = {
      modelType: 'transformer',
      parameters: {
        maxTokens: 512,
        temperature: 0.7,
        topP: 0.9,
      },
      maxExecutionTime: 30000,
      allowedOperations: ['inference', 'embeddings'],
    };

    const deployResult = await client.deployAIAgent({
      agentId: 'my-agent-001',
      modelHash: 'QmXxYyZz1234567890abcdef',
      config: agentConfig,
    });

    console.log('Agent deployed:', deployResult.agentId);
    console.log('Transaction signature:', deployResult.signature);

    // 4. Execute AI agent
    const executeResult = await client.executeAIAgent({
      agentId: 'my-agent-001',
      inputData: {
        prompt: 'Explain quantum computing in simple terms',
        maxLength: 200,
      },
      maxCompute: 100000,
    });

    console.log('Execution ID:', executeResult.executionId);
    console.log('Compute units used:', executeResult.computeUnits);
    console.log('Output:', executeResult.output);

    // 5. Get agent info
    const agentInfo = await client.getAIAgentInfo('my-agent-001');
    console.log('\nAgent Info:');
    console.log('  Model hash:', agentInfo.modelHash);
    console.log('  Owner:', agentInfo.owner);
    console.log('  Execution count:', agentInfo.executionCount);
    console.log('  Total compute used:', agentInfo.totalComputeUsed);

    // 6. Bridge operations
    console.log('\n=== Bridge Operations ===');
    
    const depositResult = await client.bridgeDeposit({
      fromChain: 'solana',
      amount: 1_000_000_000, // 1 SOL
      tokenAddress: 'So11111111111111111111111111111111111111112',
    });

    console.log('Deposit ID:', depositResult.depositId);
    console.log('Signature:', depositResult.signature);

    const withdrawResult = await client.bridgeWithdraw({
      toChain: 'solana',
      amount: 500_000_000, // 0.5 SOL
      recipient: 'YourSolanaAddressHere',
    });

    console.log('Withdrawal ID:', withdrawResult.withdrawalId);

    // 7. Network monitoring
    console.log('\n=== Network Monitoring ===');
    
    const health = await client.getHealth();
    console.log('Health status:', health.status);
    console.log('Current slot:', health.slot);

    const metrics = await client.getMetrics();
    console.log('\nMetrics:');
    console.log('  TPS:', metrics.tps);
    console.log('  Transaction count:', metrics.transactionCount);
    console.log('  Average block time:', metrics.averageBlockTime, 'ms');
    console.log('  Validator count:', metrics.validatorCount);

    const validators = await client.getValidators();
    console.log('\nActive validators:', validators.length);
    validators.slice(0, 3).forEach((v, i) => {
      console.log(`  ${i + 1}. ${v.pubkey} - Stake: ${v.stake}`);
    });

  } catch (error) {
    if (error instanceof AdeError) {
      console.error('Ade Error:', error.message);
      console.error('Error code:', error.code);
      
      switch (error.code) {
        case ErrorCode.InsufficientBalance:
          console.error('Please add more funds to your account');
          break;
        case ErrorCode.InvalidSignature:
          console.error('Transaction signature is invalid');
          break;
        case ErrorCode.ComputeBudgetExceeded:
          console.error('Reduce the compute requirements');
          break;
        default:
          console.error('Error data:', error.data);
      }
    } else {
      console.error('Unexpected error:', error);
    }
  }
}

// Advanced usage examples
async function advancedExamples() {
  const client = createAdeClient({
    rpcUrl: 'http://localhost:8899',
  });

  // Batch operations
  console.log('=== Batch Operations ===');
  const [slot, height, hash] = await Promise.all([
    client.getSlot(),
    client.getBlockHeight(),
    client.getLatestBlockhash(),
  ]);

  console.log('Slot:', slot, 'Height:', height, 'Hash:', hash.blockhash);

  // Transaction confirmation with polling
  async function confirmTransaction(signature: string): Promise<boolean> {
    const maxAttempts = 30;
    const pollInterval = 1000;

    for (let i = 0; i < maxAttempts; i++) {
      try {
        const confirmed = await client.confirmTransaction(signature);
        if (confirmed) {
          console.log('Transaction confirmed!');
          return true;
        }
      } catch (error) {
        console.log(`Attempt ${i + 1}/${maxAttempts} failed, retrying...`);
      }

      await new Promise((resolve) => setTimeout(resolve, pollInterval));
    }

    return false;
  }

  // Error handling with retry
  async function executeWithRetry<T>(
    fn: () => Promise<T>,
    maxRetries: number = 3
  ): Promise<T> {
    for (let i = 0; i < maxRetries; i++) {
      try {
        return await fn();
      } catch (error) {
        if (i === maxRetries - 1) throw error;
        
        console.log(`Retry ${i + 1}/${maxRetries}...`);
        await new Promise((resolve) => setTimeout(resolve, 1000 * (i + 1)));
      }
    }

    throw new Error('Max retries exceeded');
  }

  // Usage
  const agentInfo = await executeWithRetry(() =>
    client.getAIAgentInfo('my-agent-001')
  );
}

// Run examples
if (require.main === module) {
  main().catch(console.error);
}








