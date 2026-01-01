"""Transaction builder and utilities"""

import hashlib
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class MessageHeader:
    """Transaction message header"""
    num_required_signatures: int
    num_readonly_signed_accounts: int
    num_readonly_unsigned_accounts: int


@dataclass
class CompiledInstruction:
    """Compiled instruction"""
    program_id_index: int
    accounts: List[int]
    data: bytes


@dataclass
class Message:
    """Transaction message"""
    header: MessageHeader
    account_keys: List[bytes]
    recent_blockhash: bytes
    instructions: List[CompiledInstruction]


class Transaction:
    """Transaction object"""

    def __init__(self, message: Message):
        self.message = message
        self.signatures: List[bytes] = []

    def hash(self) -> bytes:
        """Compute transaction hash"""
        serialized = self.serialize()
        return hashlib.sha256(serialized).digest()

    def serialize(self) -> bytes:
        """Serialize transaction to bytes"""
        parts = []
        
        # Serialize signatures count
        parts.append(bytes([len(self.signatures)]))
        
        # Serialize signatures
        for sig in self.signatures:
            parts.append(sig)
        
        # Serialize message
        parts.append(self._serialize_message())
        
        return b''.join(parts)

    def _serialize_message(self) -> bytes:
        """Serialize message to bytes"""
        parts = []
        
        # Header
        parts.append(bytes([
            self.message.header.num_required_signatures,
            self.message.header.num_readonly_signed_accounts,
            self.message.header.num_readonly_unsigned_accounts,
        ]))
        
        # Account keys
        parts.append(self._encode_length(len(self.message.account_keys)))
        for key in self.message.account_keys:
            parts.append(key)
        
        # Recent blockhash
        parts.append(self.message.recent_blockhash)
        
        # Instructions
        parts.append(self._encode_length(len(self.message.instructions)))
        for instruction in self.message.instructions:
            parts.append(bytes([instruction.program_id_index]))
            parts.append(self._encode_length(len(instruction.accounts)))
            parts.append(bytes(instruction.accounts))
            parts.append(self._encode_length(len(instruction.data)))
            parts.append(instruction.data)
        
        return b''.join(parts)

    @staticmethod
    def _encode_length(length: int) -> bytes:
        """Encode length as compact-u16"""
        result = []
        while length > 0x7f:
            result.append((length & 0x7f) | 0x80)
            length >>= 7
        result.append(length)
        return bytes(result)


class TransactionBuilder:
    """Builder for constructing transactions"""

    def __init__(self):
        self.instructions: List[CompiledInstruction] = []
        self.account_keys: List[bytes] = []
        self.recent_blockhash: Optional[bytes] = None

    def add_instruction(self, instruction: CompiledInstruction) -> 'TransactionBuilder':
        """Add an instruction"""
        self.instructions.append(instruction)
        return self

    def add_account_key(self, key: bytes) -> 'TransactionBuilder':
        """Add an account key"""
        self.account_keys.append(key)
        return self

    def set_recent_blockhash(self, blockhash: bytes) -> 'TransactionBuilder':
        """Set recent blockhash"""
        self.recent_blockhash = blockhash
        return self

    def build(self) -> Transaction:
        """Build the transaction"""
        if self.recent_blockhash is None:
            raise ValueError("Recent blockhash not set")

        header = MessageHeader(
            num_required_signatures=len(self.account_keys),
            num_readonly_signed_accounts=0,
            num_readonly_unsigned_accounts=0,
        )

        message = Message(
            header=header,
            account_keys=self.account_keys,
            recent_blockhash=self.recent_blockhash,
            instructions=self.instructions,
        )

        return Transaction(message)


