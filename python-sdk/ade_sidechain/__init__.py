"""Ade Sidechain Python SDK for AI Agent Integration"""

from .client import AdeSidechainClient
from .ai_agent import AIAgent, AIAgentConfig
from .bridge import BridgeClient
from .transaction import Transaction, TransactionBuilder

__version__ = '0.1.0'

__all__ = [
    'AdeSidechainClient',
    'AIAgent',
    'AIAgentConfig',
    'BridgeClient',
    'Transaction',
    'TransactionBuilder',
]


