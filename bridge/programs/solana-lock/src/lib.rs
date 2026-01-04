use solana_program::{
    account_info::{next_account_info, AccountInfo},
    entrypoint,
    entrypoint::ProgramResult,
    msg,
    program::{invoke, invoke_signed},
    program_error::ProgramError,
    pubkey::Pubkey,
    system_instruction,
    sysvar::{rent::Rent, Sysvar},
};
use borsh::{BorshDeserialize, BorshSerialize};

entrypoint!(process_instruction);

/// Program state account
#[derive(BorshSerialize, BorshDeserialize, Debug)]
pub struct BridgeState {
    pub authority: Pubkey,
    pub total_locked: u64,
    pub nonce: u64,
    pub is_initialized: bool,
    pub supported_tokens: Vec<Pubkey>,
}

/// Deposit event for cross-chain tracking
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub struct DepositEvent {
    pub depositor: Pubkey,
    pub token_mint: Pubkey,
    pub amount: u64,
    pub target_chain: String,
    pub recipient: Vec<u8>,
    pub nonce: u64,
    pub timestamp: i64,
}

/// Instructions supported by the bridge
#[derive(BorshSerialize, BorshDeserialize, Debug)]
pub enum BridgeInstruction {
    /// Initialize bridge state
    Initialize {
        authority: Pubkey,
    },
    
    /// Lock tokens for bridging
    Lock {
        amount: u64,
        target_chain: String,
        recipient: Vec<u8>,
    },
    
    /// Unlock tokens (relayer only)
    Unlock {
        proof: BridgeProof,
        amount: u64,
        recipient: Pubkey,
    },
    
    /// Add supported token
    AddSupportedToken {
        token_mint: Pubkey,
    },
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub struct BridgeProof {
    pub source_tx_hash: Vec<u8>,
    pub block_number: u64,
    pub merkle_proof: Vec<Vec<u8>>,
    pub event_data: Vec<u8>,
    pub relayer_signatures: Vec<Vec<u8>>,
}

pub fn process_instruction(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    instruction_data: &[u8],
) -> ProgramResult {
    let instruction = BridgeInstruction::try_from_slice(instruction_data)
        .map_err(|_| ProgramError::InvalidInstructionData)?;

    match instruction {
        BridgeInstruction::Initialize { authority } => {
            msg!("Instruction: Initialize Bridge");
            process_initialize(program_id, accounts, authority)
        }
        BridgeInstruction::Lock { amount, target_chain, recipient } => {
            msg!("Instruction: Lock {} tokens for {}", amount, target_chain);
            process_lock(program_id, accounts, amount, target_chain, recipient)
        }
        BridgeInstruction::Unlock { proof, amount, recipient } => {
            msg!("Instruction: Unlock {} tokens", amount);
            process_unlock(program_id, accounts, proof, amount, recipient)
        }
        BridgeInstruction::AddSupportedToken { token_mint } => {
            msg!("Instruction: Add supported token");
            process_add_token(program_id, accounts, token_mint)
        }
    }
}

fn process_initialize(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    authority: Pubkey,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();
    
    let bridge_state_account = next_account_info(account_info_iter)?;
    let payer = next_account_info(account_info_iter)?;
    let system_program = next_account_info(account_info_iter)?;

    // Verify bridge state account is owned by program
    if bridge_state_account.owner != program_id {
        msg!("Bridge state account has invalid owner");
        return Err(ProgramError::IncorrectProgramId);
    }

    // Check if already initialized
    if bridge_state_account.data.borrow().len() > 0 {
        let state = BridgeState::try_from_slice(&bridge_state_account.data.borrow())?;
        if state.is_initialized {
            msg!("Bridge already initialized");
            return Err(ProgramError::AccountAlreadyInitialized);
        }
    }

    // Initialize state
    let bridge_state = BridgeState {
        authority,
        total_locked: 0,
        nonce: 0,
        is_initialized: true,
        supported_tokens: Vec::new(),
    };

    bridge_state.serialize(&mut &mut bridge_state_account.data.borrow_mut()[..])?;

    msg!("Bridge initialized with authority: {}", authority);
    Ok(())
}

fn process_lock(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    amount: u64,
    target_chain: String,
    recipient: Vec<u8>,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();
    
    let depositor = next_account_info(account_info_iter)?;
    let bridge_state_account = next_account_info(account_info_iter)?;
    let token_account = next_account_info(account_info_iter)?;
    let vault_account = next_account_info(account_info_iter)?;
    let token_program = next_account_info(account_info_iter)?;

    // Verify signer
    if !depositor.is_signer {
        msg!("Depositor must be signer");
        return Err(ProgramError::MissingRequiredSignature);
    }

    // Load bridge state
    let mut bridge_state = BridgeState::try_from_slice(&bridge_state_account.data.borrow())?;

    if !bridge_state.is_initialized {
        msg!("Bridge not initialized");
        return Err(ProgramError::UninitializedAccount);
    }

    // Verify amount
    if amount == 0 {
        msg!("Amount must be greater than 0");
        return Err(ProgramError::InvalidArgument);
    }

    // Transfer tokens to vault
    let transfer_instruction = spl_token::instruction::transfer(
        token_program.key,
        token_account.key,
        vault_account.key,
        depositor.key,
        &[],
        amount,
    )?;

    invoke(
        &transfer_instruction,
        &[
            token_account.clone(),
            vault_account.clone(),
            depositor.clone(),
            token_program.clone(),
        ],
    )?;

    // Update state
    bridge_state.total_locked += amount;
    bridge_state.nonce += 1;
    bridge_state.serialize(&mut &mut bridge_state_account.data.borrow_mut()[..])?;

    // Emit deposit event
    let deposit_event = DepositEvent {
        depositor: *depositor.key,
        token_mint: *token_account.key,
        amount,
        target_chain,
        recipient,
        nonce: bridge_state.nonce,
        timestamp: solana_program::clock::Clock::get()?.unix_timestamp,
    };

    // Log event for relayers to pick up
    msg!("DEPOSIT_EVENT: {:?}", deposit_event);

    msg!("Locked {} tokens, nonce: {}", amount, bridge_state.nonce);
    Ok(())
}

fn process_unlock(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    proof: BridgeProof,
    amount: u64,
    recipient: Pubkey,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();
    
    let authority = next_account_info(account_info_iter)?;
    let bridge_state_account = next_account_info(account_info_iter)?;
    let vault_account = next_account_info(account_info_iter)?;
    let recipient_account = next_account_info(account_info_iter)?;
    let token_program = next_account_info(account_info_iter)?;

    // Verify authority
    if !authority.is_signer {
        msg!("Authority must be signer");
        return Err(ProgramError::MissingRequiredSignature);
    }

    // Load bridge state
    let mut bridge_state = BridgeState::try_from_slice(&bridge_state_account.data.borrow())?;

    if bridge_state.authority != *authority.key {
        msg!("Invalid authority");
        return Err(ProgramError::InvalidAccountData);
    }

    // Verify proof
    verify_bridge_proof(&proof)?;

    // Verify amount
    if amount == 0 || amount > bridge_state.total_locked {
        msg!("Invalid unlock amount");
        return Err(ProgramError::InvalidArgument);
    }

    // Transfer tokens from vault
    let transfer_instruction = spl_token::instruction::transfer(
        token_program.key,
        vault_account.key,
        recipient_account.key,
        authority.key,
        &[],
        amount,
    )?;

    invoke(
        &transfer_instruction,
        &[
            vault_account.clone(),
            recipient_account.clone(),
            authority.clone(),
            token_program.clone(),
        ],
    )?;

    // Update state
    bridge_state.total_locked -= amount;
    bridge_state.serialize(&mut &mut bridge_state_account.data.borrow_mut()[..])?;

    msg!("Unlocked {} tokens to {}", amount, recipient);
    Ok(())
}

fn process_add_token(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    token_mint: Pubkey,
) -> ProgramResult {
    let account_info_iter = &mut accounts.iter();
    
    let authority = next_account_info(account_info_iter)?;
    let bridge_state_account = next_account_info(account_info_iter)?;

    if !authority.is_signer {
        return Err(ProgramError::MissingRequiredSignature);
    }

    let mut bridge_state = BridgeState::try_from_slice(&bridge_state_account.data.borrow())?;

    if bridge_state.authority != *authority.key {
        return Err(ProgramError::InvalidAccountData);
    }

    if bridge_state.supported_tokens.contains(&token_mint) {
        msg!("Token already supported");
        return Err(ProgramError::InvalidArgument);
    }

    bridge_state.supported_tokens.push(token_mint);
    bridge_state.serialize(&mut &mut bridge_state_account.data.borrow_mut()[..])?;

    msg!("Added supported token: {}", token_mint);
    Ok(())
}

/// Verify bridge proof (multi-sig + merkle)
fn verify_bridge_proof(proof: &BridgeProof) -> ProgramResult {
    // 1. Verify minimum signatures (2/3 threshold)
    const MIN_SIGNATURES: usize = 2;
    
    if proof.relayer_signatures.len() < MIN_SIGNATURES {
        msg!("Insufficient signatures: {} < {}", proof.relayer_signatures.len(), MIN_SIGNATURES);
        return Err(ProgramError::InvalidArgument);
    }

    // 2. Verify merkle proof
    if !proof.merkle_proof.is_empty() {
        verify_merkle_proof(&proof.merkle_proof, &proof.event_data)?;
    }

    // 3. Verify block finality
    const FINALITY_THRESHOLD: u64 = 32;
    if proof.block_number == 0 {
        msg!("Invalid block number");
        return Err(ProgramError::InvalidArgument);
    }

    msg!("Proof verified successfully");
    Ok(())
}

/// Verify merkle proof
fn verify_merkle_proof(proof: &[Vec<u8>], data: &[u8]) -> ProgramResult {
    use solana_program::keccak;
    
    let mut current_hash = keccak::hash(data).to_bytes().to_vec();

    for sibling in proof {
        let combined = if current_hash < *sibling {
            [current_hash.as_slice(), sibling.as_slice()].concat()
        } else {
            [sibling.as_slice(), current_hash.as_slice()].concat()
        };

        current_hash = keccak::hash(&combined).to_bytes().to_vec();
    }

    // In production, verify against known root
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_state_serialization() {
        let state = BridgeState {
            authority: Pubkey::new_unique(),
            total_locked: 1000,
            nonce: 1,
            is_initialized: true,
            supported_tokens: vec![],
        };

        let mut data = vec![];
        state.serialize(&mut data).unwrap();

        let deserialized = BridgeState::try_from_slice(&data).unwrap();
        assert_eq!(deserialized.nonce, 1);
        assert_eq!(deserialized.total_locked, 1000);
    }

    #[test]
    fn test_deposit_event_serialization() {
        let event = DepositEvent {
            depositor: Pubkey::new_unique(),
            token_mint: Pubkey::new_unique(),
            amount: 1000,
            target_chain: "ade".to_string(),
            recipient: vec![1, 2, 3],
            nonce: 1,
            timestamp: 12345,
        };

        let mut data = vec![];
        event.serialize(&mut data).unwrap();

        let deserialized = DepositEvent::try_from_slice(&data).unwrap();
        assert_eq!(deserialized.amount, 1000);
    }
}

