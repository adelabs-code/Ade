/// PyTorch model support (optional feature)
///
/// PRODUCTION DEPLOYMENT:
/// 
/// ## Enabling PyTorch Support
/// 
/// 1. Install libtorch:
///    - Download from https://pytorch.org/get-started/locally/
///    - Extract to a directory (e.g., /opt/libtorch)
/// 
/// 2. Set environment variables:
///    ```bash
///    export LIBTORCH=/opt/libtorch
///    export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
///    ```
/// 
/// 3. Build with feature:
///    ```bash
///    cargo build --release --features pytorch
///    ```
/// 
/// ## Alternative: Convert to ONNX
/// 
/// For production deployments without libtorch dependency, convert models to ONNX:
/// ```python
/// import torch
/// 
/// # Load your PyTorch model
/// model = torch.load('model.pt')
/// model.eval()
/// 
/// # Create dummy input matching your model's input shape
/// dummy_input = torch.randn(1, 3, 224, 224)
/// 
/// # Export to ONNX
/// torch.onnx.export(
///     model,
///     dummy_input,
///     'model.onnx',
///     opset_version=13,
///     input_names=['input'],
///     output_names=['output'],
///     dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
/// )
/// ```
/// 
/// Then use OnnxInference which requires no additional system dependencies.

#[cfg(feature = "pytorch")]
use tch::{Tensor, CModule, Kind, Device};
#[cfg(feature = "pytorch")]
use anyhow::{Result, Context};
#[cfg(feature = "pytorch")]
use std::sync::Arc;
#[cfg(feature = "pytorch")]
use tracing::{info, debug, warn};

/// GPU memory management settings
#[cfg(feature = "pytorch")]
pub struct GpuSettings {
    /// Maximum GPU memory to use (in MB, 0 = no limit)
    pub max_memory_mb: usize,
    /// Device index to use
    pub device_index: usize,
    /// Whether to allow memory growth
    pub allow_growth: bool,
}

#[cfg(feature = "pytorch")]
impl Default for GpuSettings {
    fn default() -> Self {
        Self {
            max_memory_mb: 0,
            device_index: 0,
            allow_growth: true,
        }
    }
}

#[cfg(feature = "pytorch")]
pub struct PyTorchInference {
    device: Device,
    gpu_settings: GpuSettings,
}

#[cfg(feature = "pytorch")]
impl PyTorchInference {
    /// Create new PyTorch inference engine
    pub fn new() -> Result<Self> {
        Self::with_settings(GpuSettings::default())
    }
    
    /// Create with custom GPU settings
    pub fn with_settings(settings: GpuSettings) -> Result<Self> {
        let device = if tch::Cuda::is_available() {
            let gpu_count = tch::Cuda::device_count();
            info!("CUDA available: {} GPU(s) detected", gpu_count);
            
            if settings.device_index >= gpu_count as usize {
                warn!("Requested GPU {} but only {} available, using GPU 0", 
                    settings.device_index, gpu_count);
                Device::Cuda(0)
            } else {
                info!("Using GPU {}", settings.device_index);
                Device::Cuda(settings.device_index as i32)
            }
        } else {
            info!("CUDA not available, using CPU for inference");
            info!("For GPU acceleration, install CUDA and rebuild with pytorch feature");
            Device::Cpu
        };

        Ok(Self { device, gpu_settings: settings })
    }
    
    /// Check if GPU acceleration is available
    pub fn has_gpu(&self) -> bool {
        matches!(self.device, Device::Cuda(_))
    }
    
    /// Get device info
    pub fn device_info(&self) -> String {
        match self.device {
            Device::Cpu => "CPU".to_string(),
            Device::Cuda(idx) => {
                #[cfg(feature = "pytorch")]
                {
                    // Get GPU name if available
                    format!("CUDA:{}", idx)
                }
            }
            _ => "Unknown".to_string(),
        }
    }

    /// Load TorchScript model from file
    /// 
    /// Supports models exported with:
    /// - torch.jit.script()
    /// - torch.jit.trace()
    pub fn load_model(&self, model_path: &str) -> Result<PyTorchModel> {
        info!("Loading PyTorch model from {}", model_path);
        
        // Validate file exists
        if !std::path::Path::new(model_path).exists() {
            return Err(anyhow::anyhow!("Model file not found: {}", model_path));
        }

        let module = CModule::load(model_path)
            .context(format!("Failed to load TorchScript model from {}", model_path))?;

        info!("PyTorch model loaded successfully on {}", self.device_info());

        Ok(PyTorchModel {
            module,
            device: self.device,
            model_path: model_path.to_string(),
        })
    }

    /// Load model from bytes
    pub fn load_model_from_bytes(&self, model_bytes: &[u8]) -> Result<PyTorchModel> {
        // Use secure temp directory
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join(format!("ade_model_{}.pt", 
            std::process::id()
        ));
        
        std::fs::write(&temp_path, model_bytes)
            .context("Failed to write model to temp file")?;

        let result = self.load_model(temp_path.to_str().unwrap_or("/tmp/model.pt"));

        // Clean up immediately
        let _ = std::fs::remove_file(&temp_path);
        
        result
    }
    
    /// Warm up the model with a dummy inference
    /// 
    /// This helps with first-inference latency on GPU
    pub fn warmup_model(&self, model: &PyTorchModel, input_shape: &[i64]) -> Result<()> {
        info!("Warming up model...");
        let dummy_input = Tensor::zeros(input_shape, (Kind::Float, self.device));
        let _ = model.module.forward_ts(&[dummy_input])?;
        info!("Model warmup complete");
        Ok(())
    }
}

#[cfg(feature = "pytorch")]
pub struct PyTorchModel {
    module: CModule,
    device: Device,
    model_path: String,
}

#[cfg(feature = "pytorch")]
impl PyTorchModel {
    /// Run inference on single input
    pub fn infer(&self, input: &[f32]) -> Result<Vec<f32>> {
        debug!("Running PyTorch inference on {} elements", input.len());

        // Convert to tensor and move to device
        let input_tensor = Tensor::from_slice(input)
            .to_kind(Kind::Float)
            .to(self.device);

        // Forward pass
        let output = self.module.forward_ts(&[input_tensor])
            .context("PyTorch forward pass failed")?;

        // Convert back to Vec
        let output_cpu = output.to(Device::Cpu);
        let output_vec: Vec<f32> = output_cpu.try_into()
            .context("Failed to convert output tensor to Vec")?;

        debug!("Inference complete, output size: {}", output_vec.len());
        Ok(output_vec)
    }
    
    /// Run inference with explicit input shape
    pub fn infer_shaped(&self, input: &[f32], shape: &[i64]) -> Result<Vec<f32>> {
        let input_tensor = Tensor::from_slice(input)
            .to_kind(Kind::Float)
            .reshape(shape)
            .to(self.device);

        let output = self.module.forward_ts(&[input_tensor])?;
        let output_cpu = output.to(Device::Cpu).flatten(0, -1);
        
        Ok(output_cpu.try_into()?)
    }

    /// Batch inference
    pub fn infer_batch(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(vec![]);
        }
        
        // Check all inputs have same size
        let input_size = inputs[0].len();
        if !inputs.iter().all(|i| i.len() == input_size) {
            return Err(anyhow::anyhow!("All inputs must have the same size for batch inference"));
        }
        
        debug!("Running batch inference: {} samples x {} elements", inputs.len(), input_size);
        
        // Stack into batch tensor
        let batch_data: Vec<f32> = inputs.iter().flatten().cloned().collect();
        let batch_tensor = Tensor::from_slice(&batch_data)
            .reshape(&[inputs.len() as i64, input_size as i64])
            .to_kind(Kind::Float)
            .to(self.device);
        
        let output = self.module.forward_ts(&[batch_tensor])?;
        let output_cpu = output.to(Device::Cpu);
        
        // Split back into individual results
        let output_shape = output_cpu.size();
        let output_size = if output_shape.len() > 1 { output_shape[1] as usize } else { 1 };
        
        let flat_output: Vec<f32> = output_cpu.flatten(0, -1).try_into()?;
        let results: Vec<Vec<f32>> = flat_output
            .chunks(output_size)
            .map(|chunk| chunk.to_vec())
            .collect();
        
        debug!("Batch inference complete: {} results", results.len());
        Ok(results)
    }
    
    /// Get model path
    pub fn path(&self) -> &str {
        &self.model_path
    }
}

// ===========================================================================
// STUB IMPLEMENTATIONS (when pytorch feature is disabled)
// ===========================================================================
// 
// PRODUCTION DEPLOYMENT WITHOUT PyTorch:
// 
// If you don't want to install libtorch, you have two options:
// 
// 1. RECOMMENDED: Convert PyTorch models to ONNX format
//    ONNX inference works out of the box without additional dependencies.
//    See module documentation above for conversion instructions.
// 
// 2. Use pre-built Docker images with libtorch included
//    (see deploy/docker/Dockerfile.pytorch)

#[cfg(not(feature = "pytorch"))]
pub struct PyTorchInference;

#[cfg(not(feature = "pytorch"))]
impl PyTorchInference {
    /// Create PyTorch inference engine
    /// 
    /// Returns error because pytorch feature is not enabled.
    /// 
    /// ## How to enable:
    /// 
    /// ```bash
    /// # 1. Install libtorch
    /// wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
    /// unzip libtorch*.zip -d /opt/
    /// 
    /// # 2. Set environment
    /// export LIBTORCH=/opt/libtorch
    /// export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
    /// 
    /// # 3. Build with feature
    /// cargo build --release --features pytorch
    /// ```
    /// 
    /// ## Recommended alternative:
    /// 
    /// Convert your model to ONNX and use `OnnxInference` instead.
    /// This avoids the libtorch dependency entirely.
    pub fn new() -> std::result::Result<Self, PyTorchError> {
        Err(PyTorchError::NotEnabled)
    }
    
    /// Check if PyTorch is available at runtime
    pub fn is_available() -> bool {
        false
    }
    
    /// Get setup instructions
    pub fn setup_instructions() -> &'static str {
        include_str!("../docs/pytorch_setup.md")
    }
}

/// Error type for PyTorch operations when feature is disabled
#[cfg(not(feature = "pytorch"))]
#[derive(Debug, Clone)]
pub enum PyTorchError {
    /// PyTorch feature not enabled at compile time
    NotEnabled,
    /// Model file not found or invalid
    ModelNotFound(String),
    /// Model not loaded before inference
    ModelNotLoaded,
    /// Inference execution error
    InferenceError(String),
    /// GPU/CUDA error
    DeviceError(String),
}

#[cfg(not(feature = "pytorch"))]
impl std::fmt::Display for PyTorchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PyTorchError::NotEnabled => write!(f, 
                "PyTorch support not compiled. Solutions:\n\n\
                 OPTION 1 - Enable PyTorch:\n\
                 $ export LIBTORCH=/path/to/libtorch\n\
                 $ cargo build --release --features pytorch\n\n\
                 OPTION 2 - Convert to ONNX (RECOMMENDED):\n\
                 # In Python:\n\
                 torch.onnx.export(model, dummy_input, 'model.onnx')\n\
                 # Then use OnnxInference in Rust\n\n\
                 See https://docs.ade.network/ai/pytorch-setup for details."
            ),
            PyTorchError::ModelNotFound(path) => write!(f, "Model file not found: {}", path),
            PyTorchError::ModelNotLoaded => write!(f, "No model loaded. Call load_model() first."),
            PyTorchError::InferenceError(msg) => write!(f, "Inference failed: {}", msg),
            PyTorchError::DeviceError(msg) => write!(f, "Device error: {}", msg),
        }
    }
}

#[cfg(not(feature = "pytorch"))]
impl std::error::Error for PyTorchError {}

/// Stub PyTorchModel for when feature is disabled
#[cfg(not(feature = "pytorch"))]
pub struct PyTorchModel {
    _private: (), // Prevent construction
}

#[cfg(not(feature = "pytorch"))]
impl PyTorchModel {
    /// Run inference (always fails without pytorch feature)
    pub fn infer(&self, _input: &[f32]) -> std::result::Result<Vec<f32>, PyTorchError> {
        Err(PyTorchError::NotEnabled)
    }
    
    /// Run batch inference (always fails without pytorch feature)
    pub fn infer_batch(&self, _inputs: &[Vec<f32>]) -> std::result::Result<Vec<Vec<f32>>, PyTorchError> {
        Err(PyTorchError::NotEnabled)
    }
}

/// Unified model interface
pub enum ModelBackend {
    #[cfg(feature = "pytorch")]
    PyTorch(Arc<PyTorchModel>),
    Onnx(Arc<crate::ai_inference::OnnxModel>),
}

impl ModelBackend {
    /// Run inference (unified interface)
    pub fn infer(&self, input: &[f32]) -> Result<Vec<f32>> {
        match self {
            #[cfg(feature = "pytorch")]
            ModelBackend::PyTorch(model) => model.infer(input),
            ModelBackend::Onnx(model) => {
                // Convert to ndarray and run
                let input_array = ndarray::ArrayD::from_shape_vec(
                    ndarray::IxDyn(&[input.len()]),
                    input.to_vec(),
                )?;
                
                let outputs = model.infer(input_array)?;
                if let Some(output) = outputs.first() {
                    Ok(output.iter().cloned().collect())
                } else {
                    Err(anyhow::anyhow!("No output"))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pytorch_stub() {
        #[cfg(not(feature = "pytorch"))]
        {
            let result = PyTorchInference::new();
            assert!(result.is_err());
        }
    }

    #[test]
    #[cfg(feature = "pytorch")]
    fn test_pytorch_creation() {
        let inference = PyTorchInference::new();
        assert!(inference.is_ok());
    }
}



