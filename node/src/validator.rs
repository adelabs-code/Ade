use anyhow::Result;
use ed25519_dalek::Keypair;
use std::sync::Arc;
use tracing::info;

use crate::storage::Storage;

pub struct Validator {
    keypair: Keypair,
    storage: Arc<Storage>,
}

impl Validator {
    pub fn new(keypair: Keypair, storage: Arc<Storage>) -> Result<Self> {
        Ok(Self {
            keypair,
            storage,
        })
    }

    pub async fn start(&self) -> Result<()> {
        info!("Starting validator");

        loop {
            tokio::time::sleep(tokio::time::Duration::from_millis(400)).await;
            self.produce_block().await?;
        }
    }

    async fn produce_block(&self) -> Result<()> {
        Ok(())
    }

    pub fn get_public_key(&self) -> &ed25519_dalek::PublicKey {
        &self.keypair.public
    }
}


