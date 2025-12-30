use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instruction {
    pub program_id: Vec<u8>,
    pub accounts: Vec<AccountMeta>,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
}

