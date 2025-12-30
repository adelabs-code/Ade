"""Enhanced bridge client for cross-chain operations"""

import time
import threading
from typing import Dict, Optional, Callable, List
from enum import Enum
from dataclasses import dataclass
from queue import Queue, Empty

from .client import AdeSidechainClient


class BridgeStatus(Enum):
    """Bridge operation status"""
    PENDING = 'pending'
    LOCKED = 'locked'
    RELAYED = 'relayed'
    COMPLETED = 'completed'
    FAILED = 'failed'


@dataclass
class DepositInfo:
    """Information about a bridge deposit"""
    deposit_id: str
    from_chain: str
    to_chain: str
    amount: int
    token: str
    sender: str
    recipient: str
    status: BridgeStatus
    timestamp: int
    confirmations: int = 0
    tx_hash: Optional[str] = None


@dataclass
class WithdrawalInfo:
    """Information about a bridge withdrawal"""
    withdrawal_id: str
    from_chain: str
    to_chain: str
    amount: int
    token: str
    sender: str
    recipient: str
    status: BridgeStatus
    timestamp: int
    confirmations: int = 0
    tx_hash: Optional[str] = None


class BridgeEvent:
    """Bridge event notification"""
    def __init__(
        self,
        event_type: str,
        operation_id: str,
        status: BridgeStatus,
        data: dict
    ):
        self.type = event_type
        self.id = operation_id
        self.status = status
        self.data = data
        self.timestamp = int(time.time())


class BridgeClient:
    """Enhanced client for bridge operations with monitoring"""

    def __init__(
        self,
        client: AdeSidechainClient,
        poll_interval: float = 5.0,
        confirmation_threshold: int = 32,
        timeout: int = 300
    ):
        self.client = client
        self.poll_interval = poll_interval
        self.confirmation_threshold = confirmation_threshold
        self.timeout = timeout
        
        # Monitoring state
        self._monitors: Dict[str, threading.Thread] = {}
        self._stop_events: Dict[str, threading.Event] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        self._event_queue: Queue = Queue()
        
    def deposit(
        self,
        from_chain: str,
        amount: int,
        token_address: str,
        recipient: Optional[str] = None,
        callback: Optional[Callable[[BridgeEvent], None]] = None
    ) -> DepositInfo:
        """
        Initiate a deposit from another chain to Ade sidechain
        
        Args:
            from_chain: Source chain name
            amount: Amount to deposit
            token_address: Token contract address
            recipient: Optional recipient address
            callback: Optional callback for status updates
            
        Returns:
            DepositInfo object
        """
        result = self.client.bridge_deposit(
            from_chain=from_chain,
            amount=amount,
            token_address=token_address,
        )
        
        deposit_info = DepositInfo(
            deposit_id=result['depositId'],
            from_chain=from_chain,
            to_chain='ade',
            amount=amount,
            token=token_address,
            sender='',
            recipient=recipient or '',
            status=BridgeStatus.PENDING,
            timestamp=int(time.time()),
            tx_hash=result.get('signature'),
        )
        
        # Start monitoring
        if callback:
            self.register_callback(result['depositId'], callback)
        self._start_monitoring(result['depositId'], 'deposit')
        
        return deposit_info

    def withdraw(
        self,
        to_chain: str,
        amount: int,
        recipient: str,
        token_address: Optional[str] = None,
        callback: Optional[Callable[[BridgeEvent], None]] = None
    ) -> WithdrawalInfo:
        """
        Initiate a withdrawal from Ade sidechain to another chain
        
        Args:
            to_chain: Target chain name
            amount: Amount to withdraw
            recipient: Recipient address on target chain
            token_address: Optional token address
            callback: Optional callback for status updates
            
        Returns:
            WithdrawalInfo object
        """
        result = self.client.bridge_withdraw(
            to_chain=to_chain,
            amount=amount,
            recipient=recipient,
        )
        
        withdrawal_info = WithdrawalInfo(
            withdrawal_id=result['withdrawalId'],
            from_chain='ade',
            to_chain=to_chain,
            amount=amount,
            token=token_address or '',
            sender='',
            recipient=recipient,
            status=BridgeStatus.PENDING,
            timestamp=int(time.time()),
            tx_hash=result.get('signature'),
        )
        
        # Start monitoring
        if callback:
            self.register_callback(result['withdrawalId'], callback)
        self._start_monitoring(result['withdrawalId'], 'withdraw')
        
        return withdrawal_info

    def get_deposit_info(self, deposit_id: str) -> Optional[DepositInfo]:
        """Get current deposit information"""
        try:
            info = self.client.get_bridge_status(deposit_id)
            if info:
                return DepositInfo(
                    deposit_id=info.get('depositId', deposit_id),
                    from_chain=info.get('fromChain', ''),
                    to_chain=info.get('toChain', ''),
                    amount=info.get('amount', 0),
                    token=info.get('token', ''),
                    sender=info.get('sender', ''),
                    recipient=info.get('recipient', ''),
                    status=BridgeStatus(info.get('status', 'pending')),
                    timestamp=info.get('timestamp', 0),
                    confirmations=info.get('confirmations', 0),
                    tx_hash=info.get('txHash'),
                )
        except Exception as e:
            print(f"Error getting deposit info: {e}")
        return None

    def get_withdrawal_info(self, withdrawal_id: str) -> Optional[WithdrawalInfo]:
        """Get current withdrawal information"""
        try:
            info = self.client.get_bridge_status(withdrawal_id)
            if info:
                return WithdrawalInfo(
                    withdrawal_id=info.get('withdrawalId', withdrawal_id),
                    from_chain=info.get('fromChain', ''),
                    to_chain=info.get('toChain', ''),
                    amount=info.get('amount', 0),
                    token=info.get('token', ''),
                    sender=info.get('sender', ''),
                    recipient=info.get('recipient', ''),
                    status=BridgeStatus(info.get('status', 'pending')),
                    timestamp=info.get('timestamp', 0),
                    confirmations=info.get('confirmations', 0),
                    tx_hash=info.get('txHash'),
                )
        except Exception as e:
            print(f"Error getting withdrawal info: {e}")
        return None

    def wait_for_completion(
        self,
        operation_id: str,
        operation_type: str = 'deposit',
        timeout: Optional[int] = None
    ) -> bool:
        """
        Wait for an operation to complete
        
        Args:
            operation_id: Deposit or withdrawal ID
            operation_type: 'deposit' or 'withdraw'
            timeout: Optional timeout in seconds
            
        Returns:
            True if completed successfully, False otherwise
        """
        start_time = time.time()
        timeout = timeout or self.timeout
        
        while True:
            if time.time() - start_time > timeout:
                print(f"Timeout waiting for {operation_type} {operation_id}")
                return False
            
            if operation_type == 'deposit':
                info = self.get_deposit_info(operation_id)
            else:
                info = self.get_withdrawal_info(operation_id)
            
            if not info:
                time.sleep(self.poll_interval)
                continue
            
            if info.status == BridgeStatus.COMPLETED:
                return True
            elif info.status == BridgeStatus.FAILED:
                print(f"{operation_type} failed: {operation_id}")
                return False
            
            time.sleep(self.poll_interval)

    def register_callback(
        self,
        operation_id: str,
        callback: Callable[[BridgeEvent], None]
    ):
        """Register a callback for operation status updates"""
        if operation_id not in self._callbacks:
            self._callbacks[operation_id] = []
        self._callbacks[operation_id].append(callback)

    def _start_monitoring(self, operation_id: str, operation_type: str):
        """Start monitoring an operation in background"""
        if operation_id in self._monitors:
            return
        
        stop_event = threading.Event()
        self._stop_events[operation_id] = stop_event
        
        monitor_thread = threading.Thread(
            target=self._monitor_operation,
            args=(operation_id, operation_type, stop_event),
            daemon=True
        )
        monitor_thread.start()
        self._monitors[operation_id] = monitor_thread

    def _monitor_operation(
        self,
        operation_id: str,
        operation_type: str,
        stop_event: threading.Event
    ):
        """Monitor operation status in background"""
        last_status = None
        
        while not stop_event.is_set():
            try:
                if operation_type == 'deposit':
                    info = self.get_deposit_info(operation_id)
                else:
                    info = self.get_withdrawal_info(operation_id)
                
                if info and info.status != last_status:
                    event = BridgeEvent(
                        event_type=operation_type,
                        operation_id=operation_id,
                        status=info.status,
                        data=info.__dict__
                    )
                    
                    # Trigger callbacks
                    if operation_id in self._callbacks:
                        for callback in self._callbacks[operation_id]:
                            try:
                                callback(event)
                            except Exception as e:
                                print(f"Error in callback: {e}")
                    
                    # Add to event queue
                    self._event_queue.put(event)
                    
                    last_status = info.status
                    
                    # Stop monitoring if completed or failed
                    if info.status in [BridgeStatus.COMPLETED, BridgeStatus.FAILED]:
                        break
                
            except Exception as e:
                print(f"Error monitoring {operation_type} {operation_id}: {e}")
            
            stop_event.wait(self.poll_interval)
        
        # Cleanup
        self._stop_monitoring(operation_id)

    def _stop_monitoring(self, operation_id: str):
        """Stop monitoring an operation"""
        if operation_id in self._stop_events:
            self._stop_events[operation_id].set()
            del self._stop_events[operation_id]
        
        if operation_id in self._monitors:
            del self._monitors[operation_id]
        
        if operation_id in self._callbacks:
            del self._callbacks[operation_id]

    def get_events(self, block: bool = False, timeout: Optional[float] = None) -> Optional[BridgeEvent]:
        """Get next bridge event from queue"""
        try:
            return self._event_queue.get(block=block, timeout=timeout)
        except Empty:
            return None

    def estimate_fee(
        self,
        from_chain: str,
        to_chain: str,
        amount: int
    ) -> int:
        """Estimate bridge fee"""
        base_fee = 1000
        percentage_fee = int(amount * 0.001)  # 0.1%
        return base_fee + percentage_fee

    def cleanup(self):
        """Stop all monitoring and cleanup resources"""
        for operation_id in list(self._stop_events.keys()):
            self._stop_monitoring(operation_id)


class BatchBridgeClient(BridgeClient):
    """Bridge client with batch operation support"""
    
    def batch_deposit(
        self,
        deposits: List[Dict]
    ) -> List[DepositInfo]:
        """Execute multiple deposits"""
        results = []
        for deposit in deposits:
            try:
                result = self.deposit(
                    from_chain=deposit['from_chain'],
                    amount=deposit['amount'],
                    token_address=deposit['token_address'],
                    recipient=deposit.get('recipient'),
                    callback=deposit.get('callback')
                )
                results.append(result)
            except Exception as e:
                print(f"Error in batch deposit: {e}")
        
        return results
    
    def batch_withdraw(
        self,
        withdrawals: List[Dict]
    ) -> List[WithdrawalInfo]:
        """Execute multiple withdrawals"""
        results = []
        for withdrawal in withdrawals:
            try:
                result = self.withdraw(
                    to_chain=withdrawal['to_chain'],
                    amount=withdrawal['amount'],
                    recipient=withdrawal['recipient'],
                    token_address=withdrawal.get('token_address'),
                    callback=withdrawal.get('callback')
                )
                results.append(result)
            except Exception as e:
                print(f"Error in batch withdrawal: {e}")
        
        return results
