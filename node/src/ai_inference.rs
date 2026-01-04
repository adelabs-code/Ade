use anyhow::{Result, Context};
use ndarray::{Array, ArrayD, IxDyn};
use ort::{Environment, Session, SessionBuilder, Value};
use std::path::Path;
use std::sync::Arc;
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
        // Tokenize text (simplified - in production use proper tokenizer)
        let tokens = self.tokenize_simple(text, max_length);

        // Create input array
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

    /// Simple tokenization (character-level)
    fn tokenize_simple(&self, text: &str, max_length: usize) -> Vec<i64> {
        text.chars()
            .take(max_length)
            .map(|c| c as i64)
            .collect()
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

