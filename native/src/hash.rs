use sha2::{Sha256, Digest};

pub fn hash_data(data: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}

pub fn hash_transaction_batch(transactions: &[&[u8]]) -> Vec<Vec<u8>> {
    transactions.iter()
        .map(|tx| hash_data(tx))
        .collect()
}

pub fn compute_merkle_root(hashes: &[Vec<u8>]) -> Vec<u8> {
    if hashes.is_empty() {
        return vec![0u8; 32];
    }

    if hashes.len() == 1 {
        return hashes[0].clone();
    }

    let mut current_level = hashes.to_vec();

    while current_level.len() > 1 {
        let mut next_level = Vec::new();

        for chunk in current_level.chunks(2) {
            let mut hasher = Sha256::new();
            hasher.update(&chunk[0]);
            
            if chunk.len() > 1 {
                hasher.update(&chunk[1]);
            } else {
                hasher.update(&chunk[0]);
            }

            next_level.push(hasher.finalize().to_vec());
        }

        current_level = next_level;
    }

    current_level[0].clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_data() {
        let data = b"hello world";
        let hash = hash_data(data);
        assert_eq!(hash.len(), 32);
    }

    #[test]
    fn test_merkle_root() {
        let hashes = vec![
            vec![1u8; 32],
            vec![2u8; 32],
            vec![3u8; 32],
        ];
        let root = compute_merkle_root(&hashes);
        assert_eq!(root.len(), 32);
    }
}

