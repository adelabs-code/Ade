import { AIAgentClient, AIAgentConfig, AIAgentInfo, ExecutionResult } from './ai-agent';
import { AdeSidechainClient } from './client';

/**
 * AI Agent execution queue
 */
export class AgentExecutionQueue {
  private queue: ExecutionTask[] = [];
  private processing: boolean = false;
  private maxConcurrent: number = 5;

  addTask(task: ExecutionTask): void {
    this.queue.push(task);
  }

  async processQueue(executor: (task: ExecutionTask) => Promise<any>): Promise<void> {
    if (this.processing) {
      return;
    }

    this.processing = true;

    while (this.queue.length > 0) {
      const batch = this.queue.splice(0, this.maxConcurrent);
      
      await Promise.all(
        batch.map(async (task) => {
          try {
            const result = await executor(task);
            if (task.callback) {
              task.callback(null, result);
            }
          } catch (error) {
            if (task.callback) {
              task.callback(error as Error, null);
            }
          }
        })
      );
    }

    this.processing = false;
  }

  getQueueSize(): number {
    return this.queue.length;
  }

  isProcessing(): boolean {
    return this.processing;
  }

  clear(): void {
    this.queue = [];
  }
}

export interface ExecutionTask {
  agentId: string;
  inputData: any;
  maxCompute: number;
  priority?: number;
  callback?: (error: Error | null, result: any) => void;
}

/**
 * AI Agent model registry
 */
export class ModelRegistry {
  private models: Map<string, ModelInfo> = new Map();

  registerModel(modelHash: string, info: ModelInfo): void {
    this.models.set(modelHash, info);
  }

  getModel(modelHash: string): ModelInfo | undefined {
    return this.models.get(modelHash);
  }

  listModels(): ModelInfo[] {
    return Array.from(this.models.values());
  }

  getModelsByType(type: string): ModelInfo[] {
    return Array.from(this.models.values()).filter((m) => m.type === type);
  }
}

export interface ModelInfo {
  hash: string;
  type: string;
  size: number;
  version: string;
  description?: string;
  parameters?: Record<string, any>;
}

/**
 * Execution result cache
 */
export class ExecutionCache {
  private cache: Map<string, CachedExecution> = new Map();
  private maxSize: number = 1000;
  private ttl: number = 3600000; // 1 hour

  set(key: string, result: ExecutionResult): void {
    if (this.cache.size >= this.maxSize) {
      // Evict oldest
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }

    this.cache.set(key, {
      result,
      timestamp: Date.now(),
    });
  }

  get(key: string): ExecutionResult | null {
    const cached = this.cache.get(key);
    
    if (!cached) {
      return null;
    }

    // Check TTL
    if (Date.now() - cached.timestamp > this.ttl) {
      this.cache.delete(key);
      return null;
    }

    return cached.result;
  }

  has(key: string): boolean {
    return this.get(key) !== null;
  }

  clear(): void {
    this.cache.clear();
  }

  prune(): number {
    const now = Date.now();
    let pruned = 0;

    for (const [key, cached] of this.cache.entries()) {
      if (now - cached.timestamp > this.ttl) {
        this.cache.delete(key);
        pruned++;
      }
    }

    return pruned;
  }
}

interface CachedExecution {
  result: ExecutionResult;
  timestamp: number;
}

/**
 * Advanced AI Agent client with caching and queueing
 */
export class AdvancedAIAgentClient extends AIAgentClient {
  private executionQueue: AgentExecutionQueue;
  private executionCache: ExecutionCache;
  private modelRegistry: ModelRegistry;

  constructor(client: AdeSidechainClient) {
    super(client);
    this.executionQueue = new AgentExecutionQueue();
    this.executionCache = new ExecutionCache();
    this.modelRegistry = new ModelRegistry();
  }

  /**
   * Execute with caching
   */
  async executeWithCache(
    agentId: string,
    inputData: any,
    maxCompute: number = 100000
  ): Promise<ExecutionResult> {
    const cacheKey = this.generateCacheKey(agentId, inputData);

    // Check cache
    const cached = this.executionCache.get(cacheKey);
    if (cached) {
      console.log('Using cached execution result');
      return cached;
    }

    // Execute
    const result = await this.execute(agentId, inputData, maxCompute);

    // Cache result
    this.executionCache.set(cacheKey, result);

    return result;
  }

  /**
   * Queue execution
   */
  queueExecution(
    agentId: string,
    inputData: any,
    maxCompute: number,
    callback?: (error: Error | null, result: any) => void
  ): void {
    this.executionQueue.addTask({
      agentId,
      inputData,
      maxCompute,
      callback,
    });
  }

  /**
   * Process execution queue
   */
  async processQueue(): Promise<void> {
    await this.executionQueue.processQueue(async (task) => {
      return this.execute(task.agentId, task.inputData, task.maxCompute);
    });
  }

  /**
   * Batch execute multiple tasks
   */
  async batchExecute(
    tasks: Array<{ agentId: string; inputData: any; maxCompute: number }>
  ): Promise<ExecutionResult[]> {
    return Promise.all(
      tasks.map((task) =>
        this.execute(task.agentId, task.inputData, task.maxCompute)
      )
    );
  }

  /**
   * Register model
   */
  registerModel(modelHash: string, info: ModelInfo): void {
    this.modelRegistry.registerModel(modelHash, info);
  }

  /**
   * Get model info
   */
  getModelInfo(modelHash: string): ModelInfo | undefined {
    return this.modelRegistry.getModel(modelHash);
  }

  /**
   * Clear execution cache
   */
  clearCache(): void {
    this.executionCache.clear();
  }

  /**
   * Prune expired cache entries
   */
  pruneCache(): number {
    return this.executionCache.prune();
  }

  /**
   * Get queue size
   */
  getQueueSize(): number {
    return this.executionQueue.getQueueSize();
  }

  private generateCacheKey(agentId: string, inputData: any): string {
    const dataHash = require('crypto')
      .createHash('sha256')
      .update(JSON.stringify(inputData))
      .digest('hex');
    
    return `${agentId}:${dataHash}`;
  }
}


