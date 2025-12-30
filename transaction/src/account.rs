use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    pub lamports: u64,
    pub data: Vec<u8>,
    pub owner: Vec<u8>,
    pub executable: bool,
    pub rent_epoch: u64,
}

impl Account {
    pub fn new(lamports: u64, owner: Vec<u8>) -> Self {
        Self {
            lamports,
            data: Vec::new(),
            owner,
            executable: false,
            rent_epoch: 0,
        }
    }

    pub fn with_data(mut self, data: Vec<u8>) -> Self {
        self.data = data;
        self
    }

    pub fn set_executable(mut self, executable: bool) -> Self {
        self.executable = executable;
        self
    }

    pub fn serialize(&self) -> Result<Vec<u8>, bincode::Error> {
        bincode::serialize(self)
    }

    pub fn deserialize(data: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(data)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccountState {
    Uninitialized,
    Initialized {
        owner: Vec<u8>,
        balance: u64,
    },
    AIAgent {
        agent_id: Vec<u8>,
        model_hash: Vec<u8>,
        owner: Vec<u8>,
        execution_count: u64,
        total_compute_used: u64,
    },
    Bridge {
        chain_id: String,
        total_locked: u64,
        supported_tokens: Vec<Vec<u8>>,
    },
}

impl Default for Account {
    fn default() -> Self {
        Self::new(0, vec![])
    }
}

