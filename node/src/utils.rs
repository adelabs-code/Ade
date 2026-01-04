/// Common utility functions used across the codebase

/// Get current Unix timestamp in seconds
pub fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Get current Unix timestamp in milliseconds
pub fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// Hash data using SHA256
pub fn hash_data(data: &[u8]) -> Vec<u8> {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}

/// Encode bytes to base58 string
pub fn to_base58(data: &[u8]) -> String {
    bs58::encode(data).into_string()
}

/// Decode base58 string to bytes
pub fn from_base58(s: &str) -> Result<Vec<u8>, bs58::decode::Error> {
    bs58::decode(s).into_vec()
}

