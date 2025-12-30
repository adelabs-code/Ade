"""Example: Bridge assets between chains"""

from ade_sidechain import AdeSidechainClient, BridgeClient


def main():
    # Initialize client
    client = AdeSidechainClient('http://localhost:8899')
    
    # Create bridge client
    bridge = BridgeClient(client)
    
    # Deposit from Solana to Ade sidechain
    print("Initiating bridge deposit...")
    deposit_result = bridge.deposit(
        from_chain='solana',
        amount=1000000000,  # 1 SOL in lamports
        token_address='So11111111111111111111111111111111111111112',
    )
    
    print(f"Deposit initiated!")
    print(f"Deposit ID: {deposit_result['depositId']}")
    print(f"Transaction signature: {deposit_result['signature']}")
    
    # Withdraw from Ade sidechain to Solana
    print("\nInitiating bridge withdrawal...")
    withdrawal_result = bridge.withdraw(
        to_chain='solana',
        amount=500000000,  # 0.5 SOL in lamports
        recipient='YourSolanaAddressHere...',
    )
    
    print(f"Withdrawal initiated!")
    print(f"Withdrawal ID: {withdrawal_result['withdrawalId']}")
    print(f"Transaction signature: {withdrawal_result['signature']}")


if __name__ == '__main__':
    main()

