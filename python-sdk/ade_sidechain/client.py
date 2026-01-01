"""RPC Client for Ade Sidechain"""

import json
from typing import Any, Dict, Optional
import httpx


class RpcError(Exception):
    """RPC Error"""
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"RPC Error {code}: {message}")


class AdeSidechainClient:
    """Client for interacting with Ade Sidechain RPC"""

    def __init__(self, rpc_url: str, timeout: int = 30):
        self.rpc_url = rpc_url
        self.timeout = timeout
        self._request_id = 1

    def _call(self, method: str, params: Optional[Any] = None) -> Any:
        """Make an RPC call"""
        request = {
            'jsonrpc': '2.0',
            'id': self._request_id,
            'method': method,
            'params': params,
        }
        self._request_id += 1

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self.rpc_url, json=request)
            response.raise_for_status()
            
            result = response.json()
            
            if 'error' in result:
                error = result['error']
                raise RpcError(error['code'], error['message'])
            
            return result.get('result')

    def get_slot(self) -> int:
        """Get current slot"""
        return self._call('getSlot')

    def get_block_height(self) -> int:
        """Get current block height"""
        return self._call('getBlockHeight')

    def get_latest_blockhash(self) -> Dict[str, Any]:
        """Get latest blockhash"""
        return self._call('getLatestBlockhash')

    def send_transaction(self, transaction: str) -> str:
        """Send a transaction"""
        return self._call('sendTransaction', {'transaction': transaction})

    def get_transaction(self, signature: str) -> Dict[str, Any]:
        """Get transaction details"""
        return self._call('getTransaction', {'signature': signature})

    def get_balance(self, address: str) -> int:
        """Get account balance"""
        result = self._call('getBalance', {'address': address})
        return result['value']

    def get_account_info(self, address: str) -> Dict[str, Any]:
        """Get account information"""
        return self._call('getAccountInfo', {'address': address})

    def deploy_ai_agent(
        self,
        agent_id: str,
        model_hash: str,
        config: Dict[str, Any]
    ) -> Dict[str, str]:
        """Deploy an AI agent"""
        params = {
            'agentId': agent_id,
            'modelHash': model_hash,
            'config': config,
        }
        return self._call('deployAIAgent', params)

    def execute_ai_agent(
        self,
        agent_id: str,
        input_data: Any,
        max_compute: int = 100000
    ) -> Dict[str, Any]:
        """Execute an AI agent"""
        params = {
            'agentId': agent_id,
            'inputData': input_data,
            'maxCompute': max_compute,
        }
        return self._call('executeAIAgent', params)

    def get_ai_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get AI agent information"""
        return self._call('getAIAgentInfo', {'agentId': agent_id})

    def bridge_deposit(
        self,
        from_chain: str,
        amount: int,
        token_address: str
    ) -> Dict[str, str]:
        """Initiate a bridge deposit"""
        params = {
            'fromChain': from_chain,
            'amount': amount,
            'tokenAddress': token_address,
        }
        return self._call('bridgeDeposit', params)

    def bridge_withdraw(
        self,
        to_chain: str,
        amount: int,
        recipient: str
    ) -> Dict[str, str]:
        """Initiate a bridge withdrawal"""
        params = {
            'toChain': to_chain,
            'amount': amount,
            'recipient': recipient,
        }
        return self._call('bridgeWithdraw', params)


