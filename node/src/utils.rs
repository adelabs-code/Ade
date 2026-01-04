/// Common utility functions used across the codebase

use sha2::{Sha256, Digest};

/// Get current Unix timestamp in seconds
#[inline]
pub fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Get current Unix timestamp in milliseconds
#[inline]
pub fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// Hash data using SHA256
#[inline]
pub fn hash_data(data: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}

/// Hash two pieces of data together
#[inline]
pub fn hash_pair(left: &[u8], right: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().to_vec()
}

/// Encode bytes to base58 string
#[inline]
pub fn to_base58(data: &[u8]) -> String {
    bs58::encode(data).into_string()
}

/// Decode base58 string to bytes
pub fn from_base58(s: &str) -> Result<Vec<u8>, bs58::decode::Error> {
    bs58::decode(s).into_vec()
}

/// Check if slice has valid length for pubkey
#[inline]
pub fn is_valid_pubkey_length(data: &[u8]) -> bool {
    data.len() == 32
}

/// Check if slice has valid length for signature
#[inline]
pub fn is_valid_signature_length(data: &[u8]) -> bool {
    data.len() == 64
}

/// Calculate percentage
#[inline]
pub fn calculate_percentage(part: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        (part as f64 / total as f64) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp() {
        let ts = current_timestamp();
        assert!(ts > 0);
        
        let ts_ms = current_timestamp_ms();
        assert!(ts_ms > ts * 1000);
    }

    #[test]
    fn test_hash_data() {
        let data = b"hello world";
        let hash = hash_data(data);
        assert_eq!(hash.len(), 32);
    }

    #[test]
    fn test_hash_pair() {
        let left = b"left";
        let right = b"right";
        let hash = hash_pair(left, right);
        assert_eq!(hash.len(), 32);
    }

    #[test]
    fn test_base58() {
        let data = vec![1, 2, 3, 4, 5];
        let encoded = to_base58(&data);
        let decoded = from_base58(&encoded).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn test_validation() {
        assert!(is_valid_pubkey_length(&vec![0u8; 32]));
        assert!(!is_valid_pubkey_length(&vec![0u8; 16]));
        
        assert!(is_valid_signature_length(&vec![0u8; 64]));
        assert!(!is_valid_signature_length(&vec![0u8; 32]));
    }

    #[test]
    fn test_percentage() {
        assert_eq!(calculate_percentage(50, 100), 50.0);
        assert_eq!(calculate_percentage(0, 100), 0.0);
        assert_eq!(calculate_percentage(100, 0), 0.0);
    }
}
