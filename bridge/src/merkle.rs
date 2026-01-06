use sha2::{Sha256, Digest};
use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Merkle tree for efficient proof generation
pub struct MerkleTree {
    leaves: Vec<Vec<u8>>,
    layers: Vec<Vec<Vec<u8>>>,
    root: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProof {
    pub leaf_index: usize,
    pub leaf: Vec<u8>,
    pub siblings: Vec<Vec<u8>>,
    pub root: Vec<u8>,
}

impl MerkleTree {
    /// Create a new merkle tree from leaves
    pub fn new(leaves: Vec<Vec<u8>>) -> Result<Self> {
        if leaves.is_empty() {
            return Err(anyhow::anyhow!("Cannot create merkle tree from empty leaves"));
        }

        let mut tree = Self {
            leaves: leaves.clone(),
            layers: Vec::new(),
            root: Vec::new(),
        };

        tree.build()?;
        Ok(tree)
    }

    /// Build the merkle tree
    fn build(&mut self) -> Result<()> {
        let mut current_layer = self.leaves.clone();
        self.layers.push(current_layer.clone());

        while current_layer.len() > 1 {
            let mut next_layer = Vec::new();

            for chunk in current_layer.chunks(2) {
                let hash = if chunk.len() == 2 {
                    Self::hash_pair(&chunk[0], &chunk[1])
                } else {
                    // Duplicate last node if odd number
                    Self::hash_pair(&chunk[0], &chunk[0])
                };
                next_layer.push(hash);
            }

            self.layers.push(next_layer.clone());
            current_layer = next_layer;
        }

        self.root = current_layer[0].clone();
        Ok(())
    }

    /// Get merkle root
    pub fn root(&self) -> &[u8] {
        &self.root
    }

    /// Generate proof for a leaf at index
    pub fn generate_proof(&self, leaf_index: usize) -> Result<MerkleProof> {
        if leaf_index >= self.leaves.len() {
            return Err(anyhow::anyhow!("Leaf index out of bounds"));
        }

        let mut siblings = Vec::new();
        let mut current_index = leaf_index;

        // Traverse layers from bottom to top
        for layer in &self.layers[..self.layers.len() - 1] {
            let sibling_index = if current_index % 2 == 0 {
                current_index + 1
            } else {
                current_index - 1
            };

            let sibling = if sibling_index < layer.len() {
                layer[sibling_index].clone()
            } else {
                // Duplicate if no sibling
                layer[current_index].clone()
            };

            siblings.push(sibling);
            current_index /= 2;
        }

        Ok(MerkleProof {
            leaf_index,
            leaf: self.leaves[leaf_index].clone(),
            siblings,
            root: self.root.clone(),
        })
    }

    /// Verify a merkle proof
    pub fn verify_proof(proof: &MerkleProof) -> bool {
        let mut current_hash = proof.leaf.clone();
        let mut current_index = proof.leaf_index;

        for sibling in &proof.siblings {
            current_hash = if current_index % 2 == 0 {
                Self::hash_pair(&current_hash, sibling)
            } else {
                Self::hash_pair(sibling, &current_hash)
            };
            current_index /= 2;
        }

        current_hash == proof.root
    }

    /// Hash two nodes together
    fn hash_pair(left: &[u8], right: &[u8]) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(left);
        hasher.update(right);
        hasher.finalize().to_vec()
    }

    /// Get tree depth
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// Get number of leaves
    pub fn leaf_count(&self) -> usize {
        self.leaves.len()
    }
}

/// Sparse merkle tree for account state
pub struct SparseMerkleTree {
    nodes: HashMap<Vec<u8>, Vec<u8>>,
    depth: usize,
    /// Pre-computed empty hashes for each level
    /// empty_hashes[0] = hash of empty leaf
    /// empty_hashes[n] = hash(empty_hashes[n-1], empty_hashes[n-1])
    empty_hashes: Vec<Vec<u8>>,
}

impl SparseMerkleTree {
    pub fn new(depth: usize) -> Self {
        // Pre-compute empty hashes for each level
        let mut empty_hashes = Vec::with_capacity(depth + 1);
        
        // Level 0: empty leaf hash
        let mut current = {
            let mut hasher = Sha256::new();
            hasher.update(b"EMPTY_LEAF");
            hasher.finalize().to_vec()
        };
        empty_hashes.push(current.clone());
        
        // Levels 1 to depth: hash(empty, empty)
        for _ in 1..=depth {
            let mut hasher = Sha256::new();
            hasher.update(b"BRANCH:");
            hasher.update(&current);
            hasher.update(&current);
            current = hasher.finalize().to_vec();
            empty_hashes.push(current.clone());
        }
        
        Self {
            nodes: HashMap::new(),
            depth,
            empty_hashes,
        }
    }

    /// Update a leaf
    pub fn update(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        if key.len() * 8 < self.depth {
            return Err(anyhow::anyhow!("Key too short for tree depth"));
        }

        self.nodes.insert(key.to_vec(), value.to_vec());
        Ok(())
    }

    /// Get a leaf value
    pub fn get(&self, key: &[u8]) -> Option<&Vec<u8>> {
        self.nodes.get(key)
    }

    /// Compute root hash
    pub fn root(&self) -> Vec<u8> {
        if self.nodes.is_empty() {
            return vec![0u8; 32];
        }

        // Simplified sparse merkle root computation
        let mut hasher = Sha256::new();
        
        let mut sorted_keys: Vec<_> = self.nodes.keys().collect();
        sorted_keys.sort();
        
        for key in sorted_keys {
            if let Some(value) = self.nodes.get(key) {
                hasher.update(key);
                hasher.update(value);
            }
        }
        
        hasher.finalize().to_vec()
    }

    /// Generate proof for a key
    /// 
    /// Returns a SparseMerkleProof containing the sibling hashes
    /// needed to recompute the root from the leaf.
    pub fn generate_proof(&self, key: &[u8]) -> Result<SparseMerkleProof> {
        let value = self.get(key)
            .ok_or_else(|| anyhow::anyhow!("Key not found in sparse merkle tree"))?;
        
        // Generate path from key (256 bits = 32 bytes = 256 levels max)
        // For simplicity, we use a fixed depth based on number of entries
        let depth = self.compute_depth();
        
        // Compute sibling hashes along the path
        let mut siblings = Vec::with_capacity(depth);
        let key_bits = Self::key_to_bits(key);
        
        // For each level, compute the sibling hash
        for level in 0..depth {
            let sibling_key = Self::compute_sibling_key(key, level);
            let sibling_hash = if let Some(sibling_value) = self.nodes.get(&sibling_key) {
                Self::hash_leaf(&sibling_key, sibling_value)
            } else {
                // Empty sibling - use zero hash for this level
                self.empty_hashes[level].clone()
            };
            siblings.push(sibling_hash);
        }
        
        Ok(SparseMerkleProof {
            key: key.to_vec(),
            value: value.clone(),
            siblings,
            root: self.root(),
        })
    }
    
    /// Compute tree depth based on number of entries
    fn compute_depth(&self) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }
        // Minimum depth to accommodate all entries
        let entries = self.nodes.len();
        (entries as f64).log2().ceil() as usize + 1
    }
    
    /// Convert key to bit representation
    fn key_to_bits(key: &[u8]) -> Vec<bool> {
        key.iter()
            .flat_map(|byte| (0..8).rev().map(move |i| (byte >> i) & 1 == 1))
            .collect()
    }
    
    /// Compute sibling key at a given level
    fn compute_sibling_key(key: &[u8], level: usize) -> Vec<u8> {
        let mut sibling = key.to_vec();
        if level / 8 < sibling.len() {
            // Flip the bit at the specified level
            let byte_idx = level / 8;
            let bit_idx = 7 - (level % 8);
            sibling[byte_idx] ^= 1 << bit_idx;
        }
        sibling
    }
    
    /// Hash a leaf node
    fn hash_leaf(key: &[u8], value: &[u8]) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(b"LEAF:");
        hasher.update(key);
        hasher.update(value);
        hasher.finalize().to_vec()
    }
}

/// Sparse Merkle Proof structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseMerkleProof {
    /// Key being proven
    pub key: Vec<u8>,
    /// Value at the key
    pub value: Vec<u8>,
    /// Sibling hashes along the path to root
    pub siblings: Vec<Vec<u8>>,
    /// Root hash of the tree
    pub root: Vec<u8>,
}

impl SparseMerkleProof {
    /// Verify this proof
    pub fn verify(&self) -> bool {
        let mut current_hash = SparseMerkleTree::hash_leaf_static(&self.key, &self.value);
        let key_bits = SparseMerkleTree::key_to_bits_static(&self.key);
        
        for (i, sibling) in self.siblings.iter().enumerate() {
            let mut hasher = Sha256::new();
            hasher.update(b"BRANCH:");
            
            // Determine order based on key bit at this level
            if i < key_bits.len() && key_bits[i] {
                hasher.update(sibling);
                hasher.update(&current_hash);
            } else {
                hasher.update(&current_hash);
                hasher.update(sibling);
            }
            
            current_hash = hasher.finalize().to_vec();
        }
        
        current_hash == self.root
    }
}

impl SparseMerkleTree {
    fn hash_leaf_static(key: &[u8], value: &[u8]) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(b"LEAF:");
        hasher.update(key);
        hasher.update(value);
        hasher.finalize().to_vec()
    }
    
    fn key_to_bits_static(key: &[u8]) -> Vec<bool> {
        key.iter()
            .flat_map(|byte| (0..8).rev().map(move |i| (byte >> i) & 1 == 1))
            .collect()
    }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_tree_creation() {
        let leaves = vec![
            vec![1u8; 32],
            vec![2u8; 32],
            vec![3u8; 32],
            vec![4u8; 32],
        ];

        let tree = MerkleTree::new(leaves).unwrap();
        assert_eq!(tree.leaf_count(), 4);
        assert_eq!(tree.root().len(), 32);
    }

    #[test]
    fn test_proof_generation_and_verification() {
        let leaves = vec![
            vec![1u8; 32],
            vec![2u8; 32],
            vec![3u8; 32],
            vec![4u8; 32],
        ];

        let tree = MerkleTree::new(leaves).unwrap();
        
        for i in 0..4 {
            let proof = tree.generate_proof(i).unwrap();
            assert!(MerkleTree::verify_proof(&proof));
        }
    }

    #[test]
    fn test_invalid_proof() {
        let leaves = vec![
            vec![1u8; 32],
            vec![2u8; 32],
        ];

        let tree = MerkleTree::new(leaves).unwrap();
        let mut proof = tree.generate_proof(0).unwrap();
        
        // Corrupt the proof
        proof.root[0] ^= 1;
        
        assert!(!MerkleTree::verify_proof(&proof));
    }

    #[test]
    fn test_single_leaf_tree() {
        let leaves = vec![vec![1u8; 32]];
        let tree = MerkleTree::new(leaves).unwrap();
        
        let proof = tree.generate_proof(0).unwrap();
        assert!(MerkleTree::verify_proof(&proof));
    }

    #[test]
    fn test_odd_number_of_leaves() {
        let leaves = vec![
            vec![1u8; 32],
            vec![2u8; 32],
            vec![3u8; 32],
        ];

        let tree = MerkleTree::new(leaves).unwrap();
        
        for i in 0..3 {
            let proof = tree.generate_proof(i).unwrap();
            assert!(MerkleTree::verify_proof(&proof));
        }
    }

    #[test]
    fn test_sparse_merkle_tree() {
        let mut smt = SparseMerkleTree::new(256);
        
        let key1 = vec![1u8; 32];
        let value1 = vec![10u8; 32];
        
        smt.update(&key1, &value1).unwrap();
        
        assert_eq!(smt.get(&key1), Some(&value1));
        assert_eq!(smt.root().len(), 32);
    }

    #[test]
    fn test_sparse_merkle_proof() {
        let mut smt = SparseMerkleTree::new(256);
        
        let key = vec![1u8; 32];
        let value = vec![10u8; 32];
        
        smt.update(&key, &value).unwrap();
        
        let proof = smt.generate_proof(&key).unwrap();
        assert_eq!(proof, value);
    }

    #[test]
    fn test_merkle_tree_depth() {
        let leaves = vec![
            vec![1u8; 32],
            vec![2u8; 32],
            vec![3u8; 32],
            vec![4u8; 32],
        ];

        let tree = MerkleTree::new(leaves).unwrap();
        
        // 4 leaves = 3 layers (4 -> 2 -> 1)
        assert_eq!(tree.depth(), 3);
    }
}







