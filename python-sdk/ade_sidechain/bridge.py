"""Bridge client for cross-chain operations"""

from typing import Dict
from enum import Enum
from dataclasses import dataclass
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


class BridgeClient:
    """Client for bridge operations"""

    def __init__(self, client: AdeSidechainClient):
        self.client = client

    def deposit(
        self,
        from_chain: str,
        amount: int,
        token_address: str
    ) -> Dict[str, str]:
        """Initiate a deposit from another chain"""
        return self.client.bridge_deposit(
            from_chain=from_chain,
            amount=amount,
            token_address=token_address,
        )

    def withdraw(
        self,
        to_chain: str,
        amount: int,
        recipient: str
    ) -> Dict[str, str]:
        """Initiate a withdrawal to another chain"""
        return self.client.bridge_withdraw(
            to_chain=to_chain,
            amount=amount,
            recipient=recipient,
        )

