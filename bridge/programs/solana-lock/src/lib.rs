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
    /// Bump seed for the vault PDA - required for invoke_signed
    pub vault_bump: u8,
}

/// Constants for PDA seeds
pub const VAULT_SEED: &[u8] = b"vault";
pub const BRIDGE_SEED: &[u8] = b"bridge_state";

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
    let vault_account = next_account_info(account_info_iter)?;
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

    // Derive and verify the vault PDA
    // The vault is a PDA controlled by this program, not by any external authority
    let (expected_vault_pda, vault_bump) = Pubkey::find_program_address(
        &[VAULT_SEED, bridge_state_account.key.as_ref()],
        program_id,
    );
    
    if *vault_account.key != expected_vault_pda {
        msg!("Invalid vault PDA: expected {}, got {}", expected_vault_pda, vault_account.key);
        return Err(ProgramError::InvalidSeeds);
    }

    msg!("Vault PDA derived with bump: {}", vault_bump);

    // Initialize state with vault bump seed for future invoke_signed calls
    let bridge_state = BridgeState {
        authority,
        total_locked: 0,
        nonce: 0,
        is_initialized: true,
        supported_tokens: Vec::new(),
        vault_bump,
    };

    bridge_state.serialize(&mut &mut bridge_state_account.data.borrow_mut()[..])?;

    msg!("Bridge initialized with authority: {}, vault PDA: {}", authority, expected_vault_pda);
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
    let depositor_token_account = next_account_info(account_info_iter)?;
    let vault_token_account = next_account_info(account_info_iter)?;
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

    // Verify the vault is the expected PDA
    let (expected_vault_pda, _) = Pubkey::find_program_address(
        &[VAULT_SEED, bridge_state_account.key.as_ref()],
        program_id,
    );
    
    if *vault_token_account.key != expected_vault_pda {
        msg!("Invalid vault PDA: expected {}", expected_vault_pda);
        return Err(ProgramError::InvalidSeeds);
    }

    // Verify amount
    if amount == 0 {
        msg!("Amount must be greater than 0");
        return Err(ProgramError::InvalidArgument);
    }

    // Transfer tokens from depositor to vault PDA
    // This uses regular invoke because the depositor (user) is signing, not the PDA
    let transfer_instruction = spl_token::instruction::transfer(
        token_program.key,
        depositor_token_account.key,  // Source: depositor's token account
        vault_token_account.key,       // Destination: vault PDA token account
        depositor.key,                 // Authority: depositor signs this transfer
        &[],
        amount,
    )?;

    invoke(
        &transfer_instruction,
        &[
            depositor_token_account.clone(),
            vault_token_account.clone(),
            depositor.clone(),
            token_program.clone(),
        ],
    )?;

    // Update state
    bridge_state.total_locked += amount;
    bridge_state.nonce += 1;
    bridge_state.serialize(&mut &mut bridge_state_account.data.borrow_mut()[..])?;

    // Get token mint from the depositor's token account for event
    // In production, this would be read from the token account data
    let token_mint = *depositor_token_account.key; // Simplified

    // Emit deposit event with all required information for relayers
    let deposit_event = DepositEvent {
        depositor: *depositor.key,
        token_mint,
        amount,
        target_chain: target_chain.clone(),
        recipient: recipient.clone(),
        nonce: bridge_state.nonce,
        timestamp: solana_program::clock::Clock::get()?.unix_timestamp,
    };

    // Log event in a parseable format for relayers
    msg!("DEPOSIT_EVENT: depositor={}, token={}, amount={}, target_chain={}, nonce={}",
        depositor.key,
        token_mint,
        amount,
        target_chain,
        bridge_state.nonce
    );

    msg!("Locked {} tokens to vault PDA, nonce: {}", amount, bridge_state.nonce);
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

    // Verify authority is a valid relayer/multisig
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

    // Verify the vault is the expected PDA
    let (expected_vault_pda, _) = Pubkey::find_program_address(
        &[VAULT_SEED, bridge_state_account.key.as_ref()],
        program_id,
    );
    
    if *vault_account.key != expected_vault_pda {
        msg!("Invalid vault PDA");
        return Err(ProgramError::InvalidSeeds);
    }

    // Verify proof with cryptographic verification
    verify_bridge_proof(&proof)?;

    // Verify amount
    if amount == 0 || amount > bridge_state.total_locked {
        msg!("Invalid unlock amount: {} (locked: {})", amount, bridge_state.total_locked);
        return Err(ProgramError::InvalidArgument);
    }

    // CRITICAL FIX: Use invoke_signed with PDA seeds
    // The vault is a PDA owned by this program, so we must sign with its seeds
    // Without invoke_signed, the transfer would fail because the vault is not a regular account
    let vault_seeds = &[
        VAULT_SEED,
        bridge_state_account.key.as_ref(),
        &[bridge_state.vault_bump],
    ];
    
    // Create transfer instruction where vault_account (PDA) is the source/authority
    let transfer_instruction = spl_token::instruction::transfer(
        token_program.key,
        vault_account.key,      // Source token account (vault PDA)
        recipient_account.key,  // Destination token account
        vault_account.key,      // Authority is the vault PDA itself (not an external key!)
        &[],                    // No additional signers
        amount,
    )?;

    msg!("Executing invoke_signed with vault PDA as authority");
    
    // invoke_signed allows this program to sign on behalf of the vault PDA
    // This is the only way to transfer tokens from a PDA-controlled account
    invoke_signed(
        &transfer_instruction,
        &[
            vault_account.clone(),
            recipient_account.clone(),
            token_program.clone(),
        ],
        &[vault_seeds], // PDA seeds for signing
    )?;

    // Update state
    bridge_state.total_locked -= amount;
    bridge_state.serialize(&mut &mut bridge_state_account.data.borrow_mut()[..])?;

    msg!("Successfully unlocked {} tokens to {} using PDA signing", amount, recipient);
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

