import WebSocket from 'ws';
import { EventEmitter } from 'events';

export interface SubscriptionOptions {
  reconnect?: boolean;
  reconnectDelay?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
}

export interface SlotInfo {
  slot: number;
  timestamp: number;
}

export interface AccountNotification {
  pubkey: string;
  lamports: number;
  data: any;
  owner: string;
  slot: number;
}

export interface SignatureNotification {
  signature: string;
  slot: number;
  err: any;
}

export interface BlockNotification {
  slot: number;
  hash: string;
  parentSlot: number;
  timestamp: number;
}

export type SubscriptionType = 'slot' | 'account' | 'signature' | 'block';

export class WebSocketClient extends EventEmitter {
  private ws?: WebSocket;
  private url: string;
  private options: Required<SubscriptionOptions>;
  private reconnectAttempts: number = 0;
  private subscriptions: Map<number, SubscriptionInfo> = new Map();
  private nextSubscriptionId: number = 1;
  private heartbeatTimer?: NodeJS.Timeout;
  private isConnected: boolean = false;

  constructor(url: string, options: SubscriptionOptions = {}) {
    super();
    this.url = url.replace('http://', 'ws://').replace('https://', 'wss://');
    this.options = {
      reconnect: options.reconnect !== false,
      reconnectDelay: options.reconnectDelay || 1000,
      maxReconnectAttempts: options.maxReconnectAttempts || 5,
      heartbeatInterval: options.heartbeatInterval || 30000,
    };
  }

  /**
   * Connect to WebSocket server
   */
  async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url);

      this.ws.on('open', () => {
        console.log('WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.startHeartbeat();
        this.emit('connect');
        resolve();
      });

      this.ws.on('message', (data: WebSocket.Data) => {
        this.handleMessage(data.toString());
      });

      this.ws.on('close', () => {
        console.log('WebSocket closed');
        this.isConnected = false;
        this.stopHeartbeat();
        this.emit('disconnect');

        if (this.options.reconnect) {
          this.attemptReconnect();
        }
      });

      this.ws.on('error', (error: Error) => {
        console.error('WebSocket error:', error);
        this.emit('error', error);
        reject(error);
      });
    });
  }

  /**
   * Disconnect from WebSocket
   */
  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = undefined;
    }
    this.stopHeartbeat();
  }

  /**
   * Subscribe to slot updates
   */
  slotSubscribe(callback: (info: SlotInfo) => void): number {
    const subscriptionId = this.nextSubscriptionId++;

    this.send({
      jsonrpc: '2.0',
      id: subscriptionId,
      method: 'slotSubscribe',
    });

    this.subscriptions.set(subscriptionId, {
      type: 'slot',
      callback,
    });

    return subscriptionId;
  }

  /**
   * Subscribe to account changes
   */
  accountSubscribe(pubkey: string, callback: (notification: AccountNotification) => void): number {
    const subscriptionId = this.nextSubscriptionId++;

    this.send({
      jsonrpc: '2.0',
      id: subscriptionId,
      method: 'accountSubscribe',
      params: { pubkey },
    });

    this.subscriptions.set(subscriptionId, {
      type: 'account',
      callback,
      params: { pubkey },
    });

    return subscriptionId;
  }

  /**
   * Subscribe to signature notifications
   */
  signatureSubscribe(
    signature: string,
    callback: (notification: SignatureNotification) => void
  ): number {
    const subscriptionId = this.nextSubscriptionId++;

    this.send({
      jsonrpc: '2.0',
      id: subscriptionId,
      method: 'signatureSubscribe',
      params: { signature },
    });

    this.subscriptions.set(subscriptionId, {
      type: 'signature',
      callback,
      params: { signature },
    });

    return subscriptionId;
  }

  /**
   * Subscribe to block updates
   */
  blockSubscribe(callback: (notification: BlockNotification) => void): number {
    const subscriptionId = this.nextSubscriptionId++;

    this.send({
      jsonrpc: '2.0',
      id: subscriptionId,
      method: 'blockSubscribe',
    });

    this.subscriptions.set(subscriptionId, {
      type: 'block',
      callback,
    });

    return subscriptionId;
  }

  /**
   * Unsubscribe from updates
   */
  unsubscribe(subscriptionId: number): void {
    this.send({
      jsonrpc: '2.0',
      id: subscriptionId,
      method: 'unsubscribe',
      params: { subscription_id: subscriptionId },
    });

    this.subscriptions.delete(subscriptionId);
  }

  /**
   * Unsubscribe from all
   */
  unsubscribeAll(): void {
    for (const subscriptionId of this.subscriptions.keys()) {
      this.unsubscribe(subscriptionId);
    }
  }

  /**
   * Send message
   */
  private send(data: any): void {
    if (!this.ws || !this.isConnected) {
      throw new Error('WebSocket not connected');
    }

    this.ws.send(JSON.stringify(data));
  }

  /**
   * Handle incoming message
   */
  private handleMessage(data: string): void {
    try {
      const message = JSON.parse(data);

      if (message.method) {
        // Notification
        this.handleNotification(message);
      } else if (message.result !== undefined) {
        // Subscription confirmation
        this.emit('subscribed', message.id);
      } else if (message.error) {
        // Error
        this.emit('error', new Error(message.error.message));
      }
    } catch (error) {
      console.error('Error parsing message:', error);
    }
  }

  /**
   * Handle notification
   */
  private handleNotification(message: any): void {
    const method = message.method;
    const params = message.params;

    switch (method) {
      case 'slotNotification':
        this.emit('slot', params);
        this.callSubscriptionCallbacks('slot', params);
        break;
      case 'accountNotification':
        this.emit('account', params);
        this.callSubscriptionCallbacks('account', params);
        break;
      case 'signatureNotification':
        this.emit('signature', params);
        this.callSubscriptionCallbacks('signature', params);
        break;
      case 'blockNotification':
        this.emit('block', params);
        this.callSubscriptionCallbacks('block', params);
        break;
    }
  }

  /**
   * Call subscription callbacks
   */
  private callSubscriptionCallbacks(type: SubscriptionType, data: any): void {
    for (const [_, sub] of this.subscriptions) {
      if (sub.type === type) {
        try {
          sub.callback(data);
        } catch (error) {
          console.error('Error in subscription callback:', error);
        }
      }
    }
  }

  /**
   * Start heartbeat
   */
  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (this.isConnected) {
        this.send({ type: 'ping', timestamp: Date.now() });
      }
    }, this.options.heartbeatInterval);
  }

  /**
   * Stop heartbeat
   */
  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = undefined;
    }
  }

  /**
   * Attempt reconnection
   */
  private async attemptReconnect(): Promise<void> {
    if (this.reconnectAttempts >= this.options.maxReconnectAttempts) {
      console.error('Max reconnect attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.options.reconnectDelay * this.reconnectAttempts;

    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    await this.delay(delay);

    try {
      await this.connect();
      
      // Resubscribe to all subscriptions
      for (const [id, sub] of this.subscriptions) {
        const method = `${sub.type}Subscribe`;
        this.send({
          jsonrpc: '2.0',
          id,
          method,
          params: sub.params,
        });
      }
    } catch (error) {
      console.error('Reconnect failed:', error);
    }
  }

  /**
   * Delay utility
   */
  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  /**
   * Get connection status
   */
  getConnectionStatus(): boolean {
    return this.isConnected;
  }

  /**
   * Get active subscriptions count
   */
  getSubscriptionCount(): number {
    return this.subscriptions.size;
  }
}

interface SubscriptionInfo {
  type: SubscriptionType;
  callback: (data: any) => void;
  params?: any;
}

/**
 * Subscription manager for easier usage
 */
export class SubscriptionManager {
  private wsClient: WebSocketClient;
  private subscriptionIds: Map<string, number> = new Map();

  constructor(wsUrl: string, options?: SubscriptionOptions) {
    this.wsClient = new WebSocketClient(wsUrl, options);
  }

  async connect(): Promise<void> {
    await this.wsClient.connect();
  }

  disconnect(): void {
    this.wsClient.disconnect();
  }

  /**
   * Subscribe to slots
   */
  onSlot(callback: (info: SlotInfo) => void): () => void {
    const id = this.wsClient.slotSubscribe(callback);
    this.subscriptionIds.set('slot', id);

    return () => this.wsClient.unsubscribe(id);
  }

  /**
   * Subscribe to account
   */
  onAccount(pubkey: string, callback: (notification: AccountNotification) => void): () => void {
    const id = this.wsClient.accountSubscribe(pubkey, callback);
    this.subscriptionIds.set(`account:${pubkey}`, id);

    return () => this.wsClient.unsubscribe(id);
  }

  /**
   * Subscribe to signature
   */
  onSignature(
    signature: string,
    callback: (notification: SignatureNotification) => void
  ): () => void {
    const id = this.wsClient.signatureSubscribe(signature, callback);
    this.subscriptionIds.set(`signature:${signature}`, id);

    return () => this.wsClient.unsubscribe(id);
  }

  /**
   * Subscribe to blocks
   */
  onBlock(callback: (notification: BlockNotification) => void): () => void {
    const id = this.wsClient.blockSubscribe(callback);
    this.subscriptionIds.set('block', id);

    return () => this.wsClient.unsubscribe(id);
  }

  /**
   * Remove all subscriptions
   */
  unsubscribeAll(): void {
    this.wsClient.unsubscribeAll();
    this.subscriptionIds.clear();
  }

  /**
   * Get WebSocket client
   */
  getClient(): WebSocketClient {
    return this.wsClient;
  }
}


