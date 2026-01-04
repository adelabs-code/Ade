use anyhow::{Result, Context};
use std::path::{Path, PathBuf};
use std::fs::{File, create_dir_all};
use std::io::{Write, Read};
use serde::{Serialize, Deserialize};
use tracing::{info, debug};
use flate2::Compression;
use flate2::write::GzEncoder;
use flate2::read::GzDecoder};

use crate::storage::Storage;

/// Snapshot manager for fast sync
pub struct SnapshotManager {
    snapshot_dir: PathBuf,
    compression_level: Compression,
    snapshot_interval_slots: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    pub slot: u64,
    pub block_hash: Vec<u8>,
    pub state_root: Vec<u8>,
    pub account_count: u64,
    pub total_lamports: u64,
    pub created_at: u64,
    pub file_size: u64,
    pub compressed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotManifest {
    pub snapshots: Vec<SnapshotMetadata>,
    pub latest_slot: u64,
}

impl SnapshotManager {
    pub fn new(snapshot_dir: impl AsRef<Path>, snapshot_interval_slots: u64) -> Result<Self> {
        let snapshot_dir = snapshot_dir.as_ref().to_path_buf();
        create_dir_all(&snapshot_dir)?;
        
        Ok(Self {
            snapshot_dir,
            compression_level: Compression::default(),
            snapshot_interval_slots,
        })
    }

    /// Create a snapshot at given slot
    pub fn create_snapshot(
        &self,
        slot: u64,
        storage: &Storage,
        state_root: Vec<u8>,
        block_hash: Vec<u8>,
    ) -> Result<SnapshotMetadata> {
        info!("Creating snapshot for slot {}", slot);

        let snapshot_path = self.get_snapshot_path(slot);
        let metadata_path = self.get_metadata_path(slot);

        // Collect all accounts
        let mut account_count = 0u64;
        let mut total_lamports = 0u64;
        
        // In production, this would iterate through storage
        // and collect all account data
        
        let snapshot_data = SnapshotData {
            slot,
            state_root: state_root.clone(),
            accounts: vec![], // Would contain actual accounts
        };

        // Serialize and compress
        let serialized = bincode::serialize(&snapshot_data)?;
        let mut encoder = GzEncoder::new(Vec::new(), self.compression_level);
        encoder.write_all(&serialized)?;
        let compressed = encoder.finish()?;

        // Write to file
        let mut file = File::create(&snapshot_path)?;
        file.write_all(&compressed)?;

        let metadata = SnapshotMetadata {
            slot,
            block_hash,
            state_root,
            account_count,
            total_lamports,
            created_at: current_timestamp(),
            file_size: compressed.len() as u64,
            compressed: true,
        };

        // Write metadata
        let metadata_json = serde_json::to_string_pretty(&metadata)?;
        std::fs::write(&metadata_path, metadata_json)?;

        info!("Snapshot created: {} bytes compressed", metadata.file_size);

        // Update manifest
        self.update_manifest(&metadata)?;

        Ok(metadata)
    }

    /// Load snapshot from disk
    pub fn load_snapshot(&self, slot: u64) -> Result<SnapshotData> {
        info!("Loading snapshot for slot {}", slot);

        let snapshot_path = self.get_snapshot_path(slot);
        
        if !snapshot_path.exists() {
            return Err(anyhow::anyhow!("Snapshot not found for slot {}", slot));
        }

        // Read compressed data
        let mut file = File::open(&snapshot_path)?;
        let mut compressed = Vec::new();
        file.read_to_end(&mut compressed)?;

        // Decompress
        let mut decoder = GzDecoder::new(&compressed[..]);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;

        // Deserialize
        let snapshot_data: SnapshotData = bincode::deserialize(&decompressed)?;

        info!("Snapshot loaded: {} accounts", snapshot_data.accounts.len());

        Ok(snapshot_data)
    }

    /// Restore state from snapshot
    pub fn restore_from_snapshot(&self, slot: u64, storage: &Storage) -> Result<()> {
        info!("Restoring state from snapshot at slot {}", slot);

        let snapshot_data = self.load_snapshot(slot)?;

        // Restore accounts to storage
        for (address, account_data) in &snapshot_data.accounts {
            storage.store_account(address, account_data)?;
        }

        info!("Restored {} accounts from snapshot", snapshot_data.accounts.len());

        Ok(())
    }

    /// Get latest snapshot
    pub fn get_latest_snapshot(&self) -> Result<Option<SnapshotMetadata>> {
        let manifest = self.load_manifest()?;
        
        Ok(manifest.snapshots.last().cloned())
    }

    /// List all snapshots
    pub fn list_snapshots(&self) -> Result<Vec<SnapshotMetadata>> {
        let manifest = self.load_manifest()?;
        Ok(manifest.snapshots)
    }

    /// Delete old snapshots
    pub fn prune_old_snapshots(&self, keep_count: usize) -> Result<usize> {
        let mut manifest = self.load_manifest()?;
        
        if manifest.snapshots.len() <= keep_count {
            return Ok(0);
        }

        let to_delete = manifest.snapshots.len() - keep_count;
        let deleted_snapshots: Vec<_> = manifest.snapshots.drain(..to_delete).collect();

        for snapshot in &deleted_snapshots {
            let path = self.get_snapshot_path(snapshot.slot);
            if path.exists() {
                std::fs::remove_file(&path)?;
            }
            
            let metadata_path = self.get_metadata_path(snapshot.slot);
            if metadata_path.exists() {
                std::fs::remove_file(&metadata_path)?;
            }
        }

        self.save_manifest(&manifest)?;

        info!("Pruned {} old snapshots", to_delete);
        Ok(to_delete)
    }

    /// Check if should create snapshot
    pub fn should_create_snapshot(&self, slot: u64) -> bool {
        slot % self.snapshot_interval_slots == 0
    }

    // Helper methods

    fn get_snapshot_path(&self, slot: u64) -> PathBuf {
        self.snapshot_dir.join(format!("snapshot-{}.bin.gz", slot))
    }

    fn get_metadata_path(&self, slot: u64) -> PathBuf {
        self.snapshot_dir.join(format!("snapshot-{}.json", slot))
    }

    fn get_manifest_path(&self) -> PathBuf {
        self.snapshot_dir.join("manifest.json")
    }

    fn load_manifest(&self) -> Result<SnapshotManifest> {
        let manifest_path = self.get_manifest_path();
        
        if !manifest_path.exists() {
            return Ok(SnapshotManifest {
                snapshots: Vec::new(),
                latest_slot: 0,
            });
        }

        let content = std::fs::read_to_string(&manifest_path)?;
        let manifest = serde_json::from_str(&content)?;
        
        Ok(manifest)
    }

    fn save_manifest(&self, manifest: &SnapshotManifest) -> Result<()> {
        let manifest_path = self.get_manifest_path();
        let json = serde_json::to_string_pretty(manifest)?;
        std::fs::write(&manifest_path, json)?;
        Ok(())
    }

    fn update_manifest(&self, metadata: &SnapshotMetadata) -> Result<()> {
        let mut manifest = self.load_manifest()?;
        manifest.snapshots.push(metadata.clone());
        manifest.latest_slot = metadata.slot;
        self.save_manifest(&manifest)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SnapshotData {
    pub slot: u64,
    pub state_root: Vec<u8>,
    pub accounts: Vec<(Vec<u8>, Vec<u8>)>,
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_snapshot_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let manager = SnapshotManager::new(temp_dir.path(), 10000);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_should_create_snapshot() {
        let temp_dir = TempDir::new().unwrap();
        let manager = SnapshotManager::new(temp_dir.path(), 1000).unwrap();
        
        assert!(manager.should_create_snapshot(1000));
        assert!(manager.should_create_snapshot(2000));
        assert!(!manager.should_create_snapshot(1500));
    }

    #[test]
    fn test_snapshot_creation_and_loading() {
        let temp_dir = TempDir::new().unwrap();
        let storage_dir = temp_dir.path().join("storage");
        let snapshot_dir = temp_dir.path().join("snapshots");
        
        let storage = Storage::new(storage_dir.to_str().unwrap()).unwrap();
        let manager = SnapshotManager::new(&snapshot_dir, 1000).unwrap();
        
        let metadata = manager.create_snapshot(
            1000,
            &storage,
            vec![1u8; 32],
            vec![2u8; 32],
        ).unwrap();
        
        assert_eq!(metadata.slot, 1000);
        assert!(metadata.file_size > 0);
        
        let loaded = manager.load_snapshot(1000);
        assert!(loaded.is_ok());
    }

    #[test]
    fn test_manifest_management() {
        let temp_dir = TempDir::new().unwrap();
        let storage_dir = temp_dir.path().join("storage");
        let snapshot_dir = temp_dir.path().join("snapshots");
        
        let storage = Storage::new(storage_dir.to_str().unwrap()).unwrap();
        let manager = SnapshotManager::new(&snapshot_dir, 1000).unwrap();
        
        // Create multiple snapshots
        for slot in [1000, 2000, 3000] {
            manager.create_snapshot(slot, &storage, vec![1u8; 32], vec![2u8; 32]).unwrap();
        }
        
        let snapshots = manager.list_snapshots().unwrap();
        assert_eq!(snapshots.len(), 3);
    }

    #[test]
    fn test_prune_old_snapshots() {
        let temp_dir = TempDir::new().unwrap();
        let storage_dir = temp_dir.path().join("storage");
        let snapshot_dir = temp_dir.path().join("snapshots");
        
        let storage = Storage::new(storage_dir.to_str().unwrap()).unwrap();
        let manager = SnapshotManager::new(&snapshot_dir, 1000).unwrap();
        
        // Create 5 snapshots
        for i in 1..=5 {
            manager.create_snapshot(i * 1000, &storage, vec![1u8; 32], vec![2u8; 32]).unwrap();
        }
        
        // Keep only 2 latest
        let pruned = manager.prune_old_snapshots(2).unwrap();
        assert_eq!(pruned, 3);
        
        let remaining = manager.list_snapshots().unwrap();
        assert_eq!(remaining.len(), 2);
    }
}




