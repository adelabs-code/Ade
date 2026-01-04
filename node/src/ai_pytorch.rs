/// PyTorch model support (optional feature)
///
/// Enable with: cargo build --features pytorch

#[cfg(feature = "pytorch")]
use tch::{Tensor, CModule, Kind, Device};
#[cfg(feature = "pytorch")]
use anyhow::{Result, Context};
#[cfg(feature = "pytorch")]
use std::sync::Arc;
#[cfg(feature = "pytorch")]
use tracing::{info, debug};

#[cfg(feature = "pytorch")]
pub struct PyTorchInference {
    device: Device,
}

#[cfg(feature = "pytorch")]
impl PyTorchInference {
    pub fn new() -> Result<Self> {
        let device = if tch::Cuda::is_available() {
            info!("CUDA available, using GPU");
            Device::Cuda(0)
        } else {
            info!("CUDA not available, using CPU");
            Device::Cpu
        };

        Ok(Self { device })
    }

    /// Load TorchScript model
    pub fn load_model(&self, model_path: &str) -> Result<PyTorchModel> {
        info!("Loading PyTorch model from {}", model_path);

        let module = CModule::load(model_path)?;

        info!("PyTorch model loaded successfully");

        Ok(PyTorchModel {
            module,
            device: self.device,
        })
    }

    /// Load model from bytes
    pub fn load_model_from_bytes(&self, model_bytes: &[u8]) -> Result<PyTorchModel> {
        // Write to temp file and load
        let temp_path = "/tmp/model.pt";
        std::fs::write(temp_path, model_bytes)?;

        let module = CModule::load(temp_path)?;

        // Clean up
        let _ = std::fs::remove_file(temp_path);

        Ok(PyTorchModel {
            module,
            device: self.device,
        })
    }
}

#[cfg(feature = "pytorch")]
pub struct PyTorchModel {
    module: CModule,
    device: Device,
}

#[cfg(feature = "pytorch")]
impl PyTorchModel {
    /// Run inference
    pub fn infer(&self, input: &[f32]) -> Result<Vec<f32>> {
        debug!("Running PyTorch inference");

        // Convert to tensor
        let input_tensor = Tensor::from_slice(input).to(self.device);

        // Forward pass
        let output = self.module.forward_ts(&[input_tensor])?;

        // Convert back to Vec
        let output_vec: Vec<f32> = output.try_into()?;

        debug!("Inference complete, output size: {}", output_vec.len());
        Ok(output_vec)
    }

    /// Batch inference
    pub fn infer_batch(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::new();

        for input in inputs {
            results.push(self.infer(input)?);
        }

        Ok(results)
    }
}

// Stub implementations when pytorch feature is disabled
#[cfg(not(feature = "pytorch"))]
pub struct PyTorchInference;

#[cfg(not(feature = "pytorch"))]
impl PyTorchInference {
    pub fn new() -> Result<Self, &'static str> {
        Err("PyTorch support not enabled. Build with --features pytorch")
    }
}

#[cfg(not(feature = "pytorch"))]
pub struct PyTorchModel;

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

