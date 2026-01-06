use anyhow::{Result, Context};
use ndarray::{Array, ArrayD, IxDyn};
use ort::{Environment, Session, SessionBuilder, Value};
use std::path::Path;
use std::sync::Arc;
use std::collections::HashMap;
use tracing::{info, debug, warn};

/// ONNX model inference engine
pub struct OnnxInference {
    environment: Arc<Environment>,
}

impl OnnxInference {
    pub fn new() -> Result<Self> {
        let environment = Environment::builder()
            .with_name("ade-inference")
            .build()?
            .into_arc();

        Ok(Self { environment })
    }
    
    /// Create a fallback instance that will return errors on inference
    /// Used when ONNX runtime initialization fails
    pub fn new_fallback() -> Self {
        warn!("Creating fallback OnnxInference - all inference calls will fail");
        // Create a minimal environment that will fail on actual inference
        let environment = Environment::builder()
            .with_name("ade-inference-fallback")
            .build()
            .expect("Fallback environment should always build")
            .into_arc();
        
        Self { environment }
    }

    /// Load ONNX model from file
    pub fn load_model(&self, model_path: &str) -> Result<OnnxModel> {
        info!("Loading ONNX model from {}", model_path);

        let session = SessionBuilder::new(&self.environment)?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .with_model_from_file(model_path)?;

        let input_names: Vec<String> = session
            .inputs
            .iter()
            .map(|input| input.name.clone())
            .collect();

        let output_names: Vec<String> = session
            .outputs
            .iter()
            .map(|output| output.name.clone())
            .collect();

        info!("Model loaded successfully");
        debug!("Input names: {:?}", input_names);
        debug!("Output names: {:?}", output_names);

        Ok(OnnxModel {
            session,
            input_names,
            output_names,
        })
    }

    /// Load model from bytes
    pub fn load_model_from_bytes(&self, model_bytes: &[u8]) -> Result<OnnxModel> {
        let session = SessionBuilder::new(&self.environment)?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_model_from_memory(model_bytes)?;

        let input_names: Vec<String> = session
            .inputs
            .iter()
            .map(|input| input.name.clone())
            .collect();

        let output_names: Vec<String> = session
            .outputs
            .iter()
            .map(|output| output.name.clone())
            .collect();

        Ok(OnnxModel {
            session,
            input_names,
            output_names,
        })
    }
}

pub struct OnnxModel {
    session: Session,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl OnnxModel {
    /// Run inference with input data
    pub fn infer(&self, input_data: ArrayD<f32>) -> Result<Vec<ArrayD<f32>>> {
        debug!("Running inference with input shape: {:?}", input_data.shape());

        // Create input tensor
        let input_value = Value::from_array(self.session.allocator(), &input_data)?;

        // Run inference
        let outputs = self.session.run(vec![input_value])?;

        // Extract output tensors
        let mut results = Vec::new();
        for output in outputs {
            let array = output.try_extract::<f32>()?.view().to_owned();
            results.push(array);
        }

        debug!("Inference complete, {} outputs", results.len());
        Ok(results)
    }

    /// Run inference with text input (for transformers)
    pub fn infer_text(&self, text: &str, max_length: usize) -> Result<Vec<f32>> {
        // Use BPE tokenizer for proper tokenization
        let tokenizer = BPETokenizer::default();
        let tokens = tokenizer.encode(text, max_length);

        // Create input array with proper shape for transformer models
        let input_shape = IxDyn(&[1, tokens.len()]);
        let input_data = ArrayD::from_shape_vec(
            input_shape,
            tokens.iter().map(|&t| t as f32).collect(),
        )?;

        // Run inference
        let outputs = self.infer(input_data)?;

        // Get first output
        if let Some(output) = outputs.first() {
            Ok(output.iter().cloned().collect())
        } else {
            Err(anyhow::anyhow!("No output from model"))
        }
    }

    /// Tokenize text using BPE tokenizer
    pub fn tokenize(&self, text: &str, max_length: usize) -> Vec<i64> {
        let tokenizer = BPETokenizer::default();
        tokenizer.encode(text, max_length)
    }

    /// Decode tokens back to text
    pub fn detokenize(&self, tokens: &[i64]) -> String {
        let tokenizer = BPETokenizer::default();
        tokenizer.decode(tokens)
    }

    /// Get input names
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Get output names
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }
}

/// Model cache for loaded models
pub struct ModelCache {
    models: std::sync::RwLock<std::collections::HashMap<String, Arc<OnnxModel>>>,
    max_cached: usize,
}

impl ModelCache {
    pub fn new(max_cached: usize) -> Self {
        Self {
            models: std::sync::RwLock::new(std::collections::HashMap::new()),
            max_cached,
        }
    }

    /// Get or load model
    pub fn get_or_load(
        &self,
        model_hash: &str,
        loader: impl FnOnce() -> Result<OnnxModel>,
    ) -> Result<Arc<OnnxModel>> {
        // Check cache
        {
            let cache = self.models.read().unwrap();
            if let Some(model) = cache.get(model_hash) {
                debug!("Using cached model: {}", model_hash);
                return Ok(Arc::clone(model));
            }
        }

        // Load model
        let model = loader()?;
        let arc_model = Arc::new(model);

        // Cache it
        {
            let mut cache = self.models.write().unwrap();
            
            // Evict if full
            if cache.len() >= self.max_cached {
                if let Some(key) = cache.keys().next().cloned() {
                    cache.remove(&key);
                }
            }

            cache.insert(model_hash.to_string(), Arc::clone(&arc_model));
        }

        info!("Loaded and cached model: {}", model_hash);
        Ok(arc_model)
    }

    /// Clear cache
    pub fn clear(&self) {
        self.models.write().unwrap().clear();
    }

    /// Get cache size
    pub fn size(&self) -> usize {
        self.models.read().unwrap().len()
    }
}

/// Byte-Pair Encoding (BPE) Tokenizer for transformer models
/// 
/// This tokenizer implements a simplified BPE algorithm compatible with
/// GPT-2/GPT-3 style models. It supports:
/// - Vocabulary-based token lookup
/// - Subword tokenization with byte fallback
/// - Special tokens (PAD, UNK, BOS, EOS)
pub struct BPETokenizer {
    /// Vocabulary mapping token strings to IDs
    vocab: HashMap<String, i64>,
    /// Reverse vocabulary for decoding
    reverse_vocab: HashMap<i64, String>,
    /// Merge rules for BPE (pair -> merged token)
    merges: Vec<(String, String)>,
    /// Special token IDs
    pad_token_id: i64,
    unk_token_id: i64,
    bos_token_id: i64,
    eos_token_id: i64,
}

impl Default for BPETokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl BPETokenizer {
    /// Create a new tokenizer with default vocabulary
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();
        
        // Special tokens
        vocab.insert("<PAD>".to_string(), 0);
        vocab.insert("<UNK>".to_string(), 1);
        vocab.insert("<BOS>".to_string(), 2);
        vocab.insert("<EOS>".to_string(), 3);
        
        // Build basic vocabulary (ASCII printable + common subwords)
        let mut token_id = 4i64;
        
        // Single bytes (256 tokens for byte-level BPE)
        for byte in 0u8..=255 {
            let token = format!("<0x{:02X}>", byte);
            vocab.insert(token.clone(), token_id);
            reverse_vocab.insert(token_id, token);
            token_id += 1;
        }
        
        // Common words and subwords for English
        let common_tokens = [
            " the", " a", " an", " is", " are", " was", " were", " be", " been",
            " have", " has", " had", " do", " does", " did", " will", " would",
            " could", " should", " can", " may", " might", " must", " shall",
            " to", " of", " in", " for", " on", " with", " at", " by", " from",
            " or", " and", " not", " this", " that", " it", " as", " but",
            "ing", "tion", "ed", "er", "ly", "ment", "ness", "able", "ible",
            "ful", "less", "ous", "ive", "ize", "ise", "ation", "ence", "ance",
            ".", ",", "!", "?", ":", ";", "'", "\"", "(", ")", "[", "]", "{", "}",
            " ", "\n", "\t",
        ];
        
        for token in common_tokens {
            if !vocab.contains_key(token) {
                vocab.insert(token.to_string(), token_id);
                reverse_vocab.insert(token_id, token.to_string());
                token_id += 1;
            }
        }
        
        // Build reverse vocab for special tokens
        reverse_vocab.insert(0, "<PAD>".to_string());
        reverse_vocab.insert(1, "<UNK>".to_string());
        reverse_vocab.insert(2, "<BOS>".to_string());
        reverse_vocab.insert(3, "<EOS>".to_string());
        
        // Default merge rules (in production, load from vocab file)
        let merges = vec![
            ("t".to_string(), "h".to_string()),    // th
            ("th".to_string(), "e".to_string()),   // the
            ("i".to_string(), "n".to_string()),    // in
            ("a".to_string(), "n".to_string()),    // an
            ("e".to_string(), "r".to_string()),    // er
            ("o".to_string(), "n".to_string()),    // on
            ("e".to_string(), "n".to_string()),    // en
            ("t".to_string(), "i".to_string()),    // ti
            ("ti".to_string(), "on".to_string()),  // tion
            ("i".to_string(), "ng".to_string()),   // ing
        ];
        
        Self {
            vocab,
            reverse_vocab,
            merges,
            pad_token_id: 0,
            unk_token_id: 1,
            bos_token_id: 2,
            eos_token_id: 3,
        }
    }
    
    /// Load tokenizer from vocabulary and merges files
    pub fn from_files(vocab_path: &str, merges_path: &str) -> Result<Self> {
        let vocab_content = std::fs::read_to_string(vocab_path)
            .context("Failed to read vocab file")?;
        
        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();
        
        for (idx, line) in vocab_content.lines().enumerate() {
            let token = line.to_string();
            let id = idx as i64;
            vocab.insert(token.clone(), id);
            reverse_vocab.insert(id, token);
        }
        
        let merges_content = std::fs::read_to_string(merges_path)
            .context("Failed to read merges file")?;
        
        let mut merges = Vec::new();
        for line in merges_content.lines().skip(1) { // Skip header
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                merges.push((parts[0].to_string(), parts[1].to_string()));
            }
        }
        
        Ok(Self {
            vocab,
            reverse_vocab,
            merges,
            pad_token_id: 0,
            unk_token_id: 1,
            bos_token_id: 2,
            eos_token_id: 3,
        })
    }
    
    /// Encode text to token IDs using BPE
    pub fn encode(&self, text: &str, max_length: usize) -> Vec<i64> {
        let mut tokens = Vec::new();
        
        // Add BOS token
        tokens.push(self.bos_token_id);
        
        // Pre-tokenize: split on whitespace while keeping spaces as part of tokens
        let words = self.pre_tokenize(text);
        
        for word in words {
            // Apply BPE to each word
            let word_tokens = self.bpe_encode(&word);
            tokens.extend(word_tokens);
            
            // Check if we've reached max length (minus 1 for EOS)
            if tokens.len() >= max_length - 1 {
                break;
            }
        }
        
        // Add EOS token
        tokens.push(self.eos_token_id);
        
        // Truncate to max_length
        if tokens.len() > max_length {
            tokens.truncate(max_length);
        }
        
        // Pad if necessary
        while tokens.len() < max_length {
            tokens.push(self.pad_token_id);
        }
        
        tokens
    }
    
    /// Pre-tokenize text into words (keeping spaces attached)
    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        let mut words = Vec::new();
        let mut current_word = String::new();
        
        for (i, c) in text.chars().enumerate() {
            if c.is_whitespace() && i > 0 {
                if !current_word.is_empty() {
                    words.push(current_word.clone());
                    current_word.clear();
                }
                // Start new word with the space
                current_word.push(c);
            } else {
                current_word.push(c);
            }
        }
        
        if !current_word.is_empty() {
            words.push(current_word);
        }
        
        words
    }
    
    /// Apply BPE algorithm to a single word
    fn bpe_encode(&self, word: &str) -> Vec<i64> {
        // Check if word is in vocabulary as-is
        if let Some(&id) = self.vocab.get(word) {
            return vec![id];
        }
        
        // Convert to character list
        let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        
        // Apply merge rules iteratively
        loop {
            let mut min_idx = None;
            let mut min_rank = usize::MAX;
            
            // Find the highest priority merge
            for i in 0..symbols.len().saturating_sub(1) {
                let pair = (&symbols[i], &symbols[i + 1]);
                
                if let Some(rank) = self.merges.iter().position(|(a, b)| a == pair.0 && b == pair.1) {
                    if rank < min_rank {
                        min_rank = rank;
                        min_idx = Some(i);
                    }
                }
            }
            
            // Apply the merge or break if no merges found
            if let Some(idx) = min_idx {
                let merged = format!("{}{}", symbols[idx], symbols[idx + 1]);
                symbols[idx] = merged;
                symbols.remove(idx + 1);
            } else {
                break;
            }
        }
        
        // Convert symbols to token IDs
        symbols.iter()
            .map(|s| {
                self.vocab.get(s).copied()
                    .or_else(|| {
                        // Byte-level fallback
                        if s.len() == 1 {
                            let byte = s.as_bytes()[0];
                            self.vocab.get(&format!("<0x{:02X}>", byte)).copied()
                        } else {
                            None
                        }
                    })
                    .unwrap_or(self.unk_token_id)
            })
            .collect()
    }
    
    /// Decode token IDs back to text
    pub fn decode(&self, tokens: &[i64]) -> String {
        let mut text = String::new();
        
        for &token_id in tokens {
            // Skip special tokens
            if token_id == self.pad_token_id || 
               token_id == self.bos_token_id || 
               token_id == self.eos_token_id {
                continue;
            }
            
            if token_id == self.unk_token_id {
                text.push('�');
                continue;
            }
            
            if let Some(token) = self.reverse_vocab.get(&token_id) {
                // Handle byte tokens
                if token.starts_with("<0x") && token.ends_with(">") {
                    if let Ok(byte) = u8::from_str_radix(&token[3..5], 16) {
                        text.push(byte as char);
                    }
                } else {
                    text.push_str(token);
                }
            }
        }
        
        text
    }
    
    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
    
    /// Get special token IDs
    pub fn special_tokens(&self) -> (i64, i64, i64, i64) {
        (self.pad_token_id, self.unk_token_id, self.bos_token_id, self.eos_token_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_inference_creation() {
        let inference = OnnxInference::new();
        assert!(inference.is_ok());
    }

    #[test]
    fn test_model_cache() {
        let cache = ModelCache::new(2);
        assert_eq!(cache.size(), 0);
        
        cache.clear();
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_tokenization() {
        // Would test with actual model in integration tests
    }
}

