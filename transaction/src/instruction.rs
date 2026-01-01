use serde::{Serialize, Deserialize};
use anyhow::{Result, Context};
use std::io::Cursor;
use byteorder::{ReadBytesExt, WriteBytesExt};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instruction {
    pub program_id: Vec<u8>,
    pub accounts: Vec<AccountMeta>,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AccountMeta {
    pub pubkey: Vec<u8>,
    pub is_signer: bool,
    pub is_writable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InstructionType {
    Transfer {
        from: Vec<u8>,
        to: Vec<u8>,
        amount: u64,
    },
    CreateAccount {
        owner: Vec<u8>,
        space: u64,
        lamports: u64,
    },
    AIAgentDeploy {
        agent_id: Vec<u8>,
        model_hash: Vec<u8>,
        config: Vec<u8>,
    },
    AIAgentExecute {
        agent_id: Vec<u8>,
        input_data: Vec<u8>,
        max_compute: u64,
    },
    AIAgentUpdate {
        agent_id: Vec<u8>,
        new_model_hash: Vec<u8>,
        new_config: Vec<u8>,
    },
    BridgeDeposit {
        from_chain: String,
        amount: u64,
        token_address: Vec<u8>,
    },
    BridgeWithdraw {
        to_chain: String,
        amount: u64,
        recipient: Vec<u8>,
    },
}

impl Instruction {
    pub fn new(program_id: Vec<u8>, accounts: Vec<AccountMeta>, data: Vec<u8>) -> Self {
        Self {
            program_id,
            accounts,
            data,
        }
    }

    /// Create a transfer instruction
    pub fn transfer(from: Vec<u8>, to: Vec<u8>, amount: u64, program_id: Vec<u8>) -> Self {
        let accounts = vec![
            AccountMeta {
                pubkey: from.clone(),
                is_signer: true,
                is_writable: true,
            },
            AccountMeta {
                pubkey: to.clone(),
                is_signer: false,
                is_writable: true,
            },
        ];

        let ix_type = InstructionType::Transfer { from, to, amount };
        let data = bincode::serialize(&ix_type).unwrap();

        Self::new(program_id, accounts, data)
    }

    /// Create an AI agent deployment instruction
    pub fn ai_agent_deploy(
        agent_id: Vec<u8>,
        model_hash: Vec<u8>,
        config: Vec<u8>,
        owner: Vec<u8>,
        program_id: Vec<u8>,
    ) -> Self {
        let accounts = vec![
            AccountMeta {
                pubkey: owner,
                is_signer: true,
                is_writable: true,
            },
            AccountMeta {
                pubkey: agent_id.clone(),
                is_signer: false,
                is_writable: true,
            },
        ];

        let ix_type = InstructionType::AIAgentDeploy {
            agent_id,
            model_hash,
            config,
        };
        let data = bincode::serialize(&ix_type).unwrap();

        Self::new(program_id, accounts, data)
    }

    /// Create an AI agent execution instruction
    pub fn ai_agent_execute(
        agent_id: Vec<u8>,
        input_data: Vec<u8>,
        max_compute: u64,
        caller: Vec<u8>,
        program_id: Vec<u8>,
    ) -> Self {
        let accounts = vec![
            AccountMeta {
                pubkey: caller,
                is_signer: true,
                is_writable: true,
            },
            AccountMeta {
                pubkey: agent_id.clone(),
                is_signer: false,
                is_writable: true,
            },
        ];

        let ix_type = InstructionType::AIAgentExecute {
            agent_id,
            input_data,
            max_compute,
        };
        let data = bincode::serialize(&ix_type).unwrap();

        Self::new(program_id, accounts, data)
    }

    /// Serialize instruction to bytes
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        
        // Program ID index (for now, just use 0)
        buffer.write_u8(0)?;
        
        // Account indices
        Self::write_compact_u16(&mut buffer, self.accounts.len())?;
        for (idx, account) in self.accounts.iter().enumerate() {
            buffer.write_u8(idx as u8)?;
        }
        
        // Instruction data
        Self::write_compact_u16(&mut buffer, self.data.len())?;
        buffer.extend_from_slice(&self.data);
        
        Ok(buffer)
    }

    /// Deserialize instruction from bytes
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(data);
        
        // Read program ID index
        let _program_idx = cursor.read_u8()?;
        
        // Read account indices
        let num_accounts = Self::read_compact_u16(&mut cursor)? as usize;
        let mut account_indices = Vec::with_capacity(num_accounts);
        for _ in 0..num_accounts {
            account_indices.push(cursor.read_u8()? as usize);
        }
        
        // Read instruction data
        let data_len = Self::read_compact_u16(&mut cursor)? as usize;
        let mut instruction_data = vec![0u8; data_len];
        std::io::Read::read_exact(&mut cursor, &mut instruction_data)?;
        
        // For now, create placeholder accounts
        let accounts = (0..num_accounts).map(|_| AccountMeta {
            pubkey: vec![0u8; 32],
            is_signer: false,
            is_writable: false,
        }).collect();
        
        Ok(Self {
            program_id: vec![0u8; 32],
            accounts,
            data: instruction_data,
        })
    }

    /// Get serialized size
    pub fn serialized_size(&self) -> usize {
        let mut size = 1; // program_id index
        size += Self::compact_u16_size(self.accounts.len());
        size += self.accounts.len(); // account indices
        size += Self::compact_u16_size(self.data.len());
        size += self.data.len();
        size
    }

    /// Parse instruction data as specific type
    pub fn parse_data(&self) -> Result<InstructionType> {
        bincode::deserialize(&self.data)
            .context("Failed to deserialize instruction data")
    }

    /// Get account by index
    pub fn get_account(&self, index: usize) -> Option<&AccountMeta> {
        self.accounts.get(index)
    }

    /// Get all signer accounts
    pub fn get_signers(&self) -> Vec<&AccountMeta> {
        self.accounts.iter().filter(|a| a.is_signer).collect()
    }

    /// Get all writable accounts
    pub fn get_writable_accounts(&self) -> Vec<&AccountMeta> {
        self.accounts.iter().filter(|a| a.is_writable).collect()
    }

    /// Check if instruction requires specific account
    pub fn requires_account(&self, pubkey: &[u8]) -> bool {
        self.accounts.iter().any(|a| a.pubkey == pubkey)
    }

    /// Validate instruction structure
    pub fn validate(&self) -> Result<()> {
        // Validate program ID
        if self.program_id.len() != 32 {
            return Err(anyhow::anyhow!("Invalid program ID length"));
        }

        // Validate accounts
        for account in &self.accounts {
            if account.pubkey.len() != 32 {
                return Err(anyhow::anyhow!("Invalid account pubkey length"));
            }
        }

        // Validate data
        if self.data.is_empty() {
            return Err(anyhow::anyhow!("Instruction data cannot be empty"));
        }

        Ok(())
    }

    // Helper methods for compact encoding
    fn write_compact_u16(buffer: &mut Vec<u8>, value: usize) -> Result<()> {
        let mut val = value;
        loop {
            let mut byte = (val & 0x7f) as u8;
            val >>= 7;
            if val != 0 {
                byte |= 0x80;
            }
            buffer.push(byte);
            if val == 0 {
                break;
            }
        }
        Ok(())
    }

    fn read_compact_u16(cursor: &mut Cursor<&[u8]>) -> Result<u16> {
        let mut value = 0u16;
        let mut shift = 0;
        
        loop {
            let byte = cursor.read_u8()?;
            value |= ((byte & 0x7f) as u16) << shift;
            
            if byte & 0x80 == 0 {
                break;
            }
            
            shift += 7;
            if shift >= 16 {
                return Err(anyhow::anyhow!("Compact u16 overflow"));
            }
        }
        
        Ok(value)
    }

    fn compact_u16_size(value: usize) -> usize {
        if value < 0x80 {
            1
        } else if value < 0x4000 {
            2
        } else {
            3
        }
    }
}

impl AccountMeta {
    pub fn new(pubkey: Vec<u8>, is_signer: bool, is_writable: bool) -> Self {
        Self {
            pubkey,
            is_signer,
            is_writable,
        }
    }

    /// Create a read-only account meta
    pub fn new_readonly(pubkey: Vec<u8>, is_signer: bool) -> Self {
        Self::new(pubkey, is_signer, false)
    }

    /// Create a writable account meta
    pub fn new_writable(pubkey: Vec<u8>, is_signer: bool) -> Self {
        Self::new(pubkey, is_signer, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instruction_creation() {
        let program_id = vec![1u8; 32];
        let accounts = vec![
            AccountMeta::new(vec![2u8; 32], true, true),
        ];
        let data = vec![1, 2, 3, 4];
        
        let instruction = Instruction::new(program_id.clone(), accounts, data.clone());
        
        assert_eq!(instruction.program_id, program_id);
        assert_eq!(instruction.data, data);
    }

    #[test]
    fn test_transfer_instruction() {
        let from = vec![1u8; 32];
        let to = vec![2u8; 32];
        let amount = 1000u64;
        let program_id = vec![0u8; 32];
        
        let instruction = Instruction::transfer(from.clone(), to.clone(), amount, program_id);
        
        assert_eq!(instruction.accounts.len(), 2);
        assert!(instruction.accounts[0].is_signer);
        assert!(instruction.accounts[0].is_writable);
    }

    #[test]
    fn test_instruction_serialization() {
        let program_id = vec![1u8; 32];
        let accounts = vec![
            AccountMeta::new(vec![2u8; 32], true, true),
        ];
        let data = vec![1, 2, 3, 4];
        
        let instruction = Instruction::new(program_id, accounts, data);
        let serialized = instruction.serialize().unwrap();
        
        assert!(!serialized.is_empty());
    }

    #[test]
    fn test_parse_transfer_data() {
        let from = vec![1u8; 32];
        let to = vec![2u8; 32];
        let amount = 1000u64;
        let program_id = vec![0u8; 32];
        
        let instruction = Instruction::transfer(from.clone(), to.clone(), amount, program_id);
        let parsed = instruction.parse_data().unwrap();
        
        match parsed {
            InstructionType::Transfer { from: f, to: t, amount: a } => {
                assert_eq!(f, from);
                assert_eq!(t, to);
                assert_eq!(a, amount);
            }
            _ => panic!("Wrong instruction type"),
        }
    }

    #[test]
    fn test_get_signers() {
        let program_id = vec![1u8; 32];
        let accounts = vec![
            AccountMeta::new(vec![2u8; 32], true, true),
            AccountMeta::new(vec![3u8; 32], false, true),
        ];
        let data = vec![1, 2, 3, 4];
        
        let instruction = Instruction::new(program_id, accounts, data);
        let signers = instruction.get_signers();
        
        assert_eq!(signers.len(), 1);
        assert_eq!(signers[0].pubkey, vec![2u8; 32]);
    }

    #[test]
    fn test_instruction_validation() {
        let program_id = vec![1u8; 32];
        let accounts = vec![
            AccountMeta::new(vec![2u8; 32], true, true),
        ];
        let data = vec![1, 2, 3, 4];
        
        let instruction = Instruction::new(program_id, accounts, data);
        
        assert!(instruction.validate().is_ok());
    }

    #[test]
    fn test_invalid_program_id() {
        let program_id = vec![1u8; 16]; // Wrong length
        let accounts = vec![];
        let data = vec![1];
        
        let instruction = Instruction::new(program_id, accounts, data);
        
        assert!(instruction.validate().is_err());
    }
}
