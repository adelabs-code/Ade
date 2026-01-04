use anyhow::{Result, Context};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde::{Serialize, Deserialize};
use tracing::{info, warn, debug, error};
use ndarray::{ArrayD, IxDyn};
use tokio::sync::Semaphore;

use crate::utils::{current_timestamp, hash_data, to_base58};
use crate::ai_inference::{OnnxInference, OnnxModel, ModelCache};
#[cfg(feature = "pytorch")]
use crate::ai_pytorch::{PyTorchInference, PyTorchModel};

/// AI Agent runtime environment with actual execution support
/// 
/// This runtime enforces concurrency limits using a semaphore to prevent
/// resource exhaustion when multiple inference requests arrive simultaneously.
pub struct AIRuntime {
    agents: Arc<RwLock<HashMap<Vec<u8>, AIAgentState>>>,
    execution_cache: Arc<RwLock<HashMap<Vec<u8>, ExecutionCache>>>,
    execution_history: Arc<RwLock<Vec<ExecutionRecord>>>,
    max_concurrent_executions: usize,
    max_execution_time_ms: u64,
    /// Model cache for loaded ONNX models
    model_cache: Arc<ModelCache>,
    /// ONNX inference engine
    onnx_inference: Arc<OnnxInference>,
    /// Model storage path for loading models by hash
    model_storage_path: String,
    /// Cache statistics for hit rate calculation
    cache_stats: Arc<RwLock<CacheStats>>,
    /// Semaphore for limiting concurrent AI executions
    execution_semaphore: Arc<Semaphore>,
    /// Current number of active executions (for monitoring)
    active_executions: Arc<RwLock<usize>>,
}

/// Statistics for cache hit rate calculation
#[derive(Debug, Clone, Default)]
struct CacheStats {
    hits: u64,
    misses: u64,
}

/// Storage statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub total_used_bytes: u64,
    pub max_allowed_bytes: u64,
    pub model_count: usize,
    pub max_model_size_bytes: usize,
}

/// RAII guard for tracking active executions
/// Ensures the counter is decremented when dropped
struct ExecutionGuard {
    active_executions: Arc<RwLock<usize>>,
}

impl Drop for ExecutionGuard {
    fn drop(&mut self) {
        let mut active = self.active_executions.write().unwrap();
        *active = active.saturating_sub(1);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIAgentState {
    pub agent_id: Vec<u8>,
    pub model_hash: Vec<u8>,
    pub owner: Vec<u8>,
    pub config: AgentConfig,
    pub execution_count: u64,
    pub total_compute_used: u64,
    pub created_at: u64,
    pub status: AgentStatus,
    pub last_executed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub model_type: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub max_execution_time: u64,
    pub allowed_operations: Vec<String>,
    pub compute_budget: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgentStatus {
    Active,
    Paused,
    Deleted,
}

#[derive(Debug, Clone)]
struct ExecutionCache {
    input_hash: Vec<u8>,
    output: Vec<u8>,
    compute_used: u64,
    timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExecutionRecord {
    execution_id: Vec<u8>,
    agent_id: Vec<u8>,
    timestamp: u64,
    compute_used: u64,
    success: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRequest {
    pub agent_id: Vec<u8>,
    pub input_data: Vec<u8>,
    pub max_compute: u64,
    pub caller: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub execution_id: Vec<u8>,
    pub output_data: Vec<u8>,
    pub compute_units_used: u64,
    pub logs: Vec<String>,
    pub success: bool,
    pub error: Option<String>,
}

impl AIRuntime {
    pub fn new(max_concurrent_executions: usize, max_execution_time_ms: u64) -> Self {
        Self::with_model_path(max_concurrent_executions, max_execution_time_ms, "./models".to_string())
    }

    pub fn with_model_path(max_concurrent_executions: usize, max_execution_time_ms: u64, model_storage_path: String) -> Self {
        let onnx_inference = OnnxInference::new()
            .expect("Failed to initialize ONNX inference engine");
        
        // Create semaphore with configured concurrency limit
        let semaphore = Arc::new(Semaphore::new(max_concurrent_executions));
        
        info!(
            "Initializing AI runtime with max {} concurrent executions, {} ms timeout",
            max_concurrent_executions, max_execution_time_ms
        );
        
        Self {
            agents: Arc::new(RwLock::new(HashMap::new())),
            execution_cache: Arc::new(RwLock::new(HashMap::new())),
            execution_history: Arc::new(RwLock::new(Vec::new())),
            max_concurrent_executions,
            max_execution_time_ms,
            model_cache: Arc::new(ModelCache::new(100)), // Cache up to 100 models
            onnx_inference: Arc::new(onnx_inference),
            model_storage_path,
            cache_stats: Arc::new(RwLock::new(CacheStats::default())),
            execution_semaphore: semaphore,
            active_executions: Arc::new(RwLock::new(0)),
        }
    }
    
    /// Get the current number of active AI executions
    pub fn get_active_executions(&self) -> usize {
        *self.active_executions.read().unwrap()
    }
    
    /// Get the number of available execution slots
    pub fn get_available_slots(&self) -> usize {
        self.execution_semaphore.available_permits()
    }

    /// Load a model by its hash from the model storage
    fn load_model_by_hash(&self, model_hash: &[u8]) -> Result<Arc<OnnxModel>> {
        let hash_str = to_base58(model_hash);
        
        self.model_cache.get_or_load(&hash_str, || {
            let model_path = format!("{}/{}.onnx", self.model_storage_path, hash_str);
            
            // Check if model file exists
            if std::path::Path::new(&model_path).exists() {
                self.onnx_inference.load_model(&model_path)
            } else {
                // Try loading from model bytes stored in the cache directory
                let bytes_path = format!("{}/{}.bin", self.model_storage_path, hash_str);
                if std::path::Path::new(&bytes_path).exists() {
                    let model_bytes = std::fs::read(&bytes_path)
                        .context("Failed to read model bytes")?;
                    self.onnx_inference.load_model_from_bytes(&model_bytes)
                } else {
                    Err(anyhow::anyhow!("Model not found: {}", hash_str))
                }
            }
        })
    }

    /// Maximum allowed model size (100 MB)
    /// This prevents disk exhaustion attacks from malicious uploads
    pub const MAX_MODEL_SIZE_BYTES: usize = 100 * 1024 * 1024;
    
    /// Maximum total storage for all models (10 GB)
    pub const MAX_TOTAL_STORAGE_BYTES: u64 = 10 * 1024 * 1024 * 1024;
    
    /// Store a model by its bytes and return the hash
    /// 
    /// SECURITY: Enforces size limits to prevent disk exhaustion attacks.
    /// - Individual model limit: 100 MB
    /// - Total storage limit: 10 GB
    pub fn store_model(&self, model_bytes: &[u8]) -> Result<Vec<u8>> {
        // SECURITY CHECK 1: Reject oversized models immediately
        if model_bytes.len() > Self::MAX_MODEL_SIZE_BYTES {
            error!(
                "Model upload rejected: {} bytes exceeds limit of {} bytes ({} MB)",
                model_bytes.len(),
                Self::MAX_MODEL_SIZE_BYTES,
                Self::MAX_MODEL_SIZE_BYTES / (1024 * 1024)
            );
            return Err(anyhow::anyhow!(
                "Model too large: {} MB exceeds maximum allowed size of {} MB",
                model_bytes.len() / (1024 * 1024),
                Self::MAX_MODEL_SIZE_BYTES / (1024 * 1024)
            ));
        }
        
        // Create model storage directory if it doesn't exist
        std::fs::create_dir_all(&self.model_storage_path)
            .context("Failed to create model storage directory")?;
        
        // SECURITY CHECK 2: Check total storage before writing
        let current_storage = self.get_total_storage_used()?;
        if current_storage + model_bytes.len() as u64 > Self::MAX_TOTAL_STORAGE_BYTES {
            error!(
                "Storage quota exceeded: current {} bytes + {} bytes would exceed {} bytes limit",
                current_storage,
                model_bytes.len(),
                Self::MAX_TOTAL_STORAGE_BYTES
            );
            return Err(anyhow::anyhow!(
                "Storage quota exceeded: current usage {} GB + new model {} MB would exceed {} GB limit. \
                Delete unused models to free space.",
                current_storage / (1024 * 1024 * 1024),
                model_bytes.len() / (1024 * 1024),
                Self::MAX_TOTAL_STORAGE_BYTES / (1024 * 1024 * 1024)
            ));
        }
        
        let model_hash = hash_data(model_bytes);
        let hash_str = to_base58(&model_hash);
        
        // Check if model already exists (deduplication)
        let bytes_path = format!("{}/{}.bin", self.model_storage_path, hash_str);
        if std::path::Path::new(&bytes_path).exists() {
            info!("Model already exists with hash: {} (skipping duplicate)", hash_str);
            return Ok(model_hash);
        }
        
        std::fs::write(&bytes_path, model_bytes)
            .context("Failed to store model bytes")?;
        
        info!(
            "Stored model with hash: {} ({} bytes, total storage: {} MB)",
            hash_str,
            model_bytes.len(),
            (current_storage + model_bytes.len() as u64) / (1024 * 1024)
        );
        
        Ok(model_hash)
    }
    
    /// Get total storage used by all models
    fn get_total_storage_used(&self) -> Result<u64> {
        let mut total: u64 = 0;
        
        if !std::path::Path::new(&self.model_storage_path).exists() {
            return Ok(0);
        }
        
        for entry in std::fs::read_dir(&self.model_storage_path)
            .context("Failed to read model storage directory")?
        {
            if let Ok(entry) = entry {
                if let Ok(metadata) = entry.metadata() {
                    total += metadata.len();
                }
            }
        }
        
        Ok(total)
    }
    
    /// Delete a model by hash (for cleanup)
    pub fn delete_model(&self, model_hash: &[u8]) -> Result<()> {
        let hash_str = to_base58(model_hash);
        
        let onnx_path = format!("{}/{}.onnx", self.model_storage_path, hash_str);
        let bin_path = format!("{}/{}.bin", self.model_storage_path, hash_str);
        
        let mut deleted = false;
        
        if std::path::Path::new(&onnx_path).exists() {
            std::fs::remove_file(&onnx_path)
                .context("Failed to delete ONNX model")?;
            deleted = true;
        }
        
        if std::path::Path::new(&bin_path).exists() {
            std::fs::remove_file(&bin_path)
                .context("Failed to delete binary model")?;
            deleted = true;
        }
        
        if deleted {
            info!("Deleted model with hash: {}", hash_str);
            // Evict from cache
            self.model_cache.evict(&hash_str);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Model not found: {}", hash_str))
        }
    }
    
    /// Get storage statistics
    pub fn get_storage_stats(&self) -> Result<StorageStats> {
        let total_used = self.get_total_storage_used()?;
        let model_count = if std::path::Path::new(&self.model_storage_path).exists() {
            std::fs::read_dir(&self.model_storage_path)
                .map(|entries| entries.count())
                .unwrap_or(0)
        } else {
            0
        };
        
        Ok(StorageStats {
            total_used_bytes: total_used,
            max_allowed_bytes: Self::MAX_TOTAL_STORAGE_BYTES,
            model_count,
            max_model_size_bytes: Self::MAX_MODEL_SIZE_BYTES,
        })
    }

    pub fn deploy_agent(
        &self,
        agent_id: Vec<u8>,
        model_hash: Vec<u8>,
        owner: Vec<u8>,
        config: AgentConfig,
    ) -> Result<()> {
        let mut agents = self.agents.write().unwrap();
        
        if agents.contains_key(&agent_id) {
            return Err(anyhow::anyhow!("Agent already exists"));
        }

        let agent_state = AIAgentState {
            agent_id: agent_id.clone(),
            model_hash,
            owner,
            config,
            execution_count: 0,
            total_compute_used: 0,
            created_at: current_timestamp(),
            status: AgentStatus::Active,
            last_executed: None,
        };

        agents.insert(agent_id.clone(), agent_state);
        info!("Deployed AI agent: {}", to_base58(&agent_id));

        Ok(())
    }

    pub async fn execute_agent(&self, request: ExecutionRequest) -> Result<ExecutionResult> {
        let agent = {
            let agents = self.agents.read().unwrap();
            agents.get(&request.agent_id)
                .ok_or_else(|| anyhow::anyhow!("Agent not found"))?
                .clone()
        };

        if agent.status != AgentStatus::Active {
            return Err(anyhow::anyhow!("Agent is not active"));
        }

        if request.max_compute > agent.config.compute_budget {
            return Err(anyhow::anyhow!(
                "Requested compute {} exceeds budget {}",
                request.max_compute,
                agent.config.compute_budget
            ));
        }

        let input_hash = hash_data(&request.input_data);
        
        // Check cache for existing execution result (no semaphore needed for cache reads)
        if let Some(cached) = self.get_cached_execution(&input_hash) {
            self.record_cache_hit();
            info!("Cache HIT - Using cached execution result");
            return Ok(ExecutionResult {
                execution_id: self.generate_execution_id(&request.agent_id, &input_hash),
                output_data: cached.output,
                compute_units_used: cached.compute_used,
                logs: vec![
                    "Cache HIT: Using cached result".to_string(),
                    format!("Cache hit rate: {:.2}%", self.calculate_cache_hit_rate() * 100.0),
                ],
                success: true,
                error: None,
            });
        }
        
        // Cache miss - will need to execute
        self.record_cache_miss();

        // Acquire semaphore permit to limit concurrent executions
        // This prevents resource exhaustion when many requests arrive simultaneously
        let available_before = self.execution_semaphore.available_permits();
        debug!(
            "Waiting for execution slot ({}/{} available)",
            available_before, self.max_concurrent_executions
        );
        
        // Try to acquire permit with timeout to prevent indefinite blocking
        let permit = match tokio::time::timeout(
            std::time::Duration::from_secs(30),
            self.execution_semaphore.acquire()
        ).await {
            Ok(Ok(permit)) => permit,
            Ok(Err(_)) => {
                return Err(anyhow::anyhow!(
                    "Failed to acquire execution slot: semaphore closed"
                ));
            }
            Err(_) => {
                warn!(
                    "Execution queue timeout: no slot available after 30s ({} active executions)",
                    self.get_active_executions()
                );
                return Err(anyhow::anyhow!(
                    "Execution timeout: all {} slots busy for 30+ seconds",
                    self.max_concurrent_executions
                ));
            }
        };
        
        // Track active executions
        {
            let mut active = self.active_executions.write().unwrap();
            *active += 1;
            debug!(
                "Acquired execution slot ({}/{} now active)",
                *active, self.max_concurrent_executions
            );
        }
        
        // Create guard to decrement active count on completion
        // The permit is also kept in scope to be released when guard is dropped
        let _guard = ExecutionGuard {
            active_executions: self.active_executions.clone(),
        };
        let _permit = permit; // Keep permit in scope until function returns

        let start_time = std::time::Instant::now();
        let mut logs = Vec::new();
        
        logs.push(format!("Starting execution for agent: {}", to_base58(&request.agent_id)));
        logs.push(format!("Model type: {}", agent.config.model_type));
        logs.push(format!("Input size: {} bytes", request.input_data.len()));
        logs.push(format!("Max compute: {}", request.max_compute));
        logs.push(format!(
            "Concurrency: {}/{} slots in use",
            self.get_active_executions(), self.max_concurrent_executions
        ));

        // Execute AI model
        let (output, compute_used) = self.execute_ai_model(
            &agent,
            &request.input_data,
            request.max_compute
        ).await?;
        
        let execution_time = start_time.elapsed().as_millis() as u64;
        logs.push(format!("Execution completed in {}ms", execution_time));
        logs.push(format!("Compute units used: {}", compute_used));
        logs.push(format!("Output size: {} bytes", output.len()));

        // Update agent state
        {
            let mut agents = self.agents.write().unwrap();
            if let Some(agent_state) = agents.get_mut(&request.agent_id) {
                agent_state.execution_count += 1;
                agent_state.total_compute_used += compute_used;
                agent_state.last_executed = Some(current_timestamp());
            }
        }

        // Cache result
        self.cache_execution(input_hash.clone(), output.clone(), compute_used);

        // Record execution
        let execution_id = self.generate_execution_id(&request.agent_id, &input_hash);
        self.record_execution(&execution_id, &request.agent_id, compute_used, true);

        Ok(ExecutionResult {
            execution_id,
            output_data: output,
            compute_units_used: compute_used,
            logs,
            success: true,
            error: None,
        })
    }

    /// Execute AI model - actual implementation with real inference
    async fn execute_ai_model(
        &self,
        agent: &AIAgentState,
        input_data: &[u8],
        max_compute: u64,
    ) -> Result<(Vec<u8>, u64)> {
        let mut compute_used = 0u64;
        
        // 1. Model loading cost
        const MODEL_LOAD_COST: u64 = 10_000;
        compute_used += MODEL_LOAD_COST;
        
        if compute_used > max_compute {
            return Err(anyhow::anyhow!("Compute budget exceeded during model loading"));
        }

        // 2. Input processing cost (per byte)
        let input_processing_cost = (input_data.len() as u64) * 10;
        compute_used += input_processing_cost;
        
        if compute_used > max_compute {
            return Err(anyhow::anyhow!("Compute budget exceeded during input processing"));
        }

        // 3. Load the actual model from cache - REQUIRED, no fallback
        let model = self.load_model_by_hash(&agent.model_hash)
            .map_err(|e| {
                error!(
                    "Model not found for agent {}: {}. Model hash: {}",
                    to_base58(&agent.agent_id),
                    e,
                    to_base58(&agent.model_hash)
                );
                anyhow::anyhow!(
                    "ModelNotFound: Cannot execute agent without model. \
                    Model hash '{}' not found in storage path '{}'. \
                    Please ensure the model is uploaded before executing the agent.",
                    to_base58(&agent.model_hash),
                    self.model_storage_path
                )
            })?;
        
        // 4. Execute using the REAL model - no simulation fallback
        // Pass Arc<OnnxModel> for thread-safe sharing in spawn_blocking
        let (output, execution_cost) = self.execute_with_real_model(
            model, 
            input_data, 
            &agent.config, 
            max_compute - compute_used
        ).await?;

        compute_used += execution_cost;
        
        if compute_used > max_compute {
            return Err(anyhow::anyhow!("Compute budget exceeded during execution"));
        }

        // 5. Output processing cost
        let output_processing_cost = (output.len() as u64) * 5;
        compute_used += output_processing_cost;

        Ok((output, compute_used.min(max_compute)))
    }

    /// Execute inference with a real loaded ONNX model
    /// 
    /// CRITICAL: Uses tokio::task::spawn_blocking to offload heavy CPU-bound
    /// inference to a dedicated thread pool. This prevents blocking the async
    /// runtime's event loop, which would otherwise cause:
    /// - Missed network heartbeats/pings
    /// - Delayed block/vote processing
    /// - Potential consensus timeouts
    /// - Node appearing "dead" to peers
    async fn execute_with_real_model(
        &self,
        model: Arc<OnnxModel>,
        input_data: &[u8],
        config: &AgentConfig,
        remaining_compute: u64,
    ) -> Result<(Vec<u8>, u64)> {
        let start_time = std::time::Instant::now();
        
        // Clone data for the blocking task
        let model_type = config.model_type.clone();
        let max_tokens = config.parameters.get("maxTokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(512) as usize;
        let input_bytes = input_data.to_vec();
        
        // The model is already Arc-wrapped, clone the Arc for the blocking task
        let model_clone = model;
        
        // CRITICAL FIX: Offload CPU-intensive inference to blocking thread pool
        // This prevents the async runtime from being blocked during model execution
        let inference_result = tokio::task::spawn_blocking(move || {
            let inference_start = std::time::Instant::now();
            
            let output_data = match model_type.as_str() {
                "transformer" | "embeddings" => {
                    // Text input - use infer_text
                    let input_text = String::from_utf8_lossy(&input_bytes);
                    
                    let output = model_clone.infer_text(&input_text, max_tokens)?;
                    
                    Ok::<Vec<u8>, anyhow::Error>(serde_json::json!({
                        "model": model_type,
                        "input_length": input_text.len(),
                        "output_shape": [output.len()],
                        "output": output,
                        "timestamp": std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        "inference_type": "real",
                        "execution_thread": "blocking_pool"
                    }).to_string().into_bytes())
                }
                "cnn" | "rnn" | _ => {
                    // Binary/tensor input - convert to float array
                    let input_floats: Vec<f32> = input_bytes.iter()
                        .map(|&b| b as f32 / 255.0) // Normalize to [0, 1]
                        .collect();
                    
                    let input_shape = IxDyn(&[1, input_floats.len()]);
                    let input_array = ArrayD::from_shape_vec(input_shape, input_floats)
                        .context("Failed to create input tensor")?;
                    
                    let outputs = model_clone.infer(input_array)?;
                    
                    let output_vecs: Vec<Vec<f32>> = outputs.iter()
                        .map(|arr| arr.iter().cloned().collect())
                        .collect();
                    
                    Ok(serde_json::json!({
                        "model": model_type,
                        "input_size": input_bytes.len(),
                        "output_count": outputs.len(),
                        "outputs": output_vecs,
                        "timestamp": std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        "inference_type": "real",
                        "execution_thread": "blocking_pool"
                    }).to_string().into_bytes())
                }
            };
            
            let inference_time_us = inference_start.elapsed().as_micros() as u64;
            output_data.map(|data| (data, inference_time_us))
        })
        .await
        .context("Inference task panicked")?
        .context("Inference execution failed")?;
        
        let (output_data, inference_time_us) = inference_result;
        let execution_time_ms = start_time.elapsed().as_millis() as u64;
        
        // Compute cost based on actual execution time
        // 1 compute unit per microsecond
        let compute_cost = inference_time_us.min(remaining_compute);
        
        info!(
            "Model inference completed in {}ms ({}μs in blocking pool), compute cost: {}",
            execution_time_ms, inference_time_us, compute_cost
        );
        
        Ok((output_data, compute_cost))
    }

    // NOTE: Simulated model execution functions have been REMOVED
    // All AI inference now requires a real ONNX model to be loaded.
    // This prevents fake results from being passed off as real inference.
    // 
    // If you need test models, create actual minimal ONNX models
    // and store them using store_model().

    pub fn get_agent_info(&self, agent_id: &[u8]) -> Option<AIAgentState> {
        let agents = self.agents.read().unwrap();
        agents.get(agent_id).cloned()
    }

    pub fn update_agent(
        &self,
        agent_id: &[u8],
        caller: &[u8],
        new_config: AgentConfig,
    ) -> Result<()> {
        let mut agents = self.agents.write().unwrap();
        
        let agent = agents.get_mut(agent_id)
            .ok_or_else(|| anyhow::anyhow!("Agent not found"))?;

        if agent.owner != caller {
            return Err(anyhow::anyhow!("Unauthorized: not the owner"));
        }

        agent.config = new_config;
        info!("Updated agent configuration");

        Ok(())
    }

    pub fn pause_agent(&self, agent_id: &[u8], caller: &[u8]) -> Result<()> {
        self.update_agent_status(agent_id, caller, AgentStatus::Paused)
    }

    pub fn resume_agent(&self, agent_id: &[u8], caller: &[u8]) -> Result<()> {
        self.update_agent_status(agent_id, caller, AgentStatus::Active)
    }

    pub fn delete_agent(&self, agent_id: &[u8], caller: &[u8]) -> Result<()> {
        self.update_agent_status(agent_id, caller, AgentStatus::Deleted)
    }

    fn update_agent_status(&self, agent_id: &[u8], caller: &[u8], status: AgentStatus) -> Result<()> {
        let mut agents = self.agents.write().unwrap();
        
        let agent = agents.get_mut(agent_id)
            .ok_or_else(|| anyhow::anyhow!("Agent not found"))?;

        if agent.owner != caller {
            return Err(anyhow::anyhow!("Unauthorized"));
        }

        agent.status = status;
        Ok(())
    }

    pub fn list_agents_by_owner(&self, owner: &[u8]) -> Vec<AIAgentState> {
        let agents = self.agents.read().unwrap();
        
        agents.values()
            .filter(|a| a.owner == owner && a.status != AgentStatus::Deleted)
            .cloned()
            .collect()
    }

    pub fn get_active_agents(&self) -> Vec<AIAgentState> {
        let agents = self.agents.read().unwrap();
        
        agents.values()
            .filter(|a| a.status == AgentStatus::Active)
            .cloned()
            .collect()
    }

    pub fn get_execution_history(&self, agent_id: &[u8], limit: usize) -> Vec<ExecutionRecord> {
        let history = self.execution_history.read().unwrap();
        
        history.iter()
            .filter(|r| r.agent_id == agent_id)
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    pub fn get_stats(&self) -> RuntimeStats {
        let agents = self.agents.read().unwrap();
        
        let total_agents = agents.len();
        let active_agents = agents.values().filter(|a| a.status == AgentStatus::Active).count();
        let total_executions: u64 = agents.values().map(|a| a.execution_count).sum();
        let total_compute: u64 = agents.values().map(|a| a.total_compute_used).sum();
        
        let cache = self.execution_cache.read().unwrap();
        let history = self.execution_history.read().unwrap();
        
        RuntimeStats {
            total_agents,
            active_agents,
            total_executions,
            total_compute_used: total_compute,
            cache_size: cache.len(),
            cache_hit_rate: self.calculate_cache_hit_rate(),
            total_execution_records: history.len(),
        }
    }

    /// Calculate the cache hit rate based on actual cache statistics
    fn calculate_cache_hit_rate(&self) -> f64 {
        let stats = self.cache_stats.read().unwrap();
        let total = stats.hits + stats.misses;
        
        if total == 0 {
            return 0.0;
        }
        
        (stats.hits as f64) / (total as f64)
    }
    
    /// Record a cache hit
    fn record_cache_hit(&self) {
        let mut stats = self.cache_stats.write().unwrap();
        stats.hits += 1;
    }
    
    /// Record a cache miss
    fn record_cache_miss(&self) {
        let mut stats = self.cache_stats.write().unwrap();
        stats.misses += 1;
    }
    
    /// Reset cache statistics
    pub fn reset_cache_stats(&self) {
        let mut stats = self.cache_stats.write().unwrap();
        stats.hits = 0;
        stats.misses = 0;
    }

    fn generate_execution_id(&self, agent_id: &[u8], input_hash: &[u8]) -> Vec<u8> {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(b"execution");
        hasher.update(agent_id);
        hasher.update(input_hash);
        hasher.update(&current_timestamp().to_le_bytes());
        hasher.finalize().to_vec()
    }

    fn get_cached_execution(&self, input_hash: &[u8]) -> Option<ExecutionCache> {
        let cache = self.execution_cache.read().unwrap();
        cache.get(input_hash).cloned()
    }

    fn cache_execution(&self, input_hash: Vec<u8>, output: Vec<u8>, compute_used: u64) {
        const MAX_CACHE_SIZE: usize = 1000;
        const CACHE_PRUNE_SIZE: usize = 100;
        
        let mut cache = self.execution_cache.write().unwrap();
        
        if cache.len() >= MAX_CACHE_SIZE {
            let oldest_keys: Vec<_> = cache.iter()
                .take(CACHE_PRUNE_SIZE)
                .map(|(k, _)| k.clone())
                .collect();
            
            for key in oldest_keys {
                cache.remove(&key);
            }
        }
        
        cache.insert(input_hash, ExecutionCache {
            input_hash: vec![],
            output,
            compute_used,
            timestamp: current_timestamp(),
        });
    }

    fn record_execution(&self, execution_id: &[u8], agent_id: &[u8], compute_used: u64, success: bool) {
        const MAX_HISTORY_SIZE: usize = 10_000;
        const HISTORY_PRUNE_SIZE: usize = 1_000;
        
        let mut history = self.execution_history.write().unwrap();
        
        if history.len() >= MAX_HISTORY_SIZE {
            history.drain(0..HISTORY_PRUNE_SIZE);
        }
        
        history.push(ExecutionRecord {
            execution_id: execution_id.to_vec(),
            agent_id: agent_id.to_vec(),
            timestamp: current_timestamp(),
            compute_used,
            success,
        });
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeStats {
    pub total_agents: usize,
    pub active_agents: usize,
    pub total_executions: u64,
    pub total_compute_used: u64,
    pub cache_size: usize,
    pub cache_hit_rate: f64,
    pub total_execution_records: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_config() -> AgentConfig {
        AgentConfig {
            model_type: "transformer".to_string(),
            parameters: HashMap::new(),
            max_execution_time: 30000,
            allowed_operations: vec!["inference".to_string()],
            compute_budget: 100000,
        }
    }

    fn create_test_runtime() -> (AIRuntime, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let runtime = AIRuntime::with_model_path(
            10, 
            30000, 
            temp_dir.path().to_str().unwrap().to_string()
        );
        (runtime, temp_dir)
    }

    #[test]
    fn test_deploy_agent() {
        let (runtime, _temp_dir) = create_test_runtime();
        
        let agent_id = vec![1u8; 32];
        let model_hash = vec![2u8; 32];
        let owner = vec![3u8; 32];
        let config = create_test_config();
        
        let result = runtime.deploy_agent(agent_id.clone(), model_hash, owner, config);
        assert!(result.is_ok());
        
        let info = runtime.get_agent_info(&agent_id);
        assert!(info.is_some());
    }

    #[tokio::test]
    async fn test_execute_transformer() {
        let (runtime, _temp_dir) = create_test_runtime();
        
        let agent_id = vec![1u8; 32];
        let owner = vec![3u8; 32];
        let mut config = create_test_config();
        config.parameters.insert("maxTokens".to_string(), serde_json::json!(256));
        
        runtime.deploy_agent(agent_id.clone(), vec![2u8; 32], owner.clone(), config).unwrap();
        
        let request = ExecutionRequest {
            agent_id: agent_id.clone(),
            input_data: b"Hello, this is a test input for transformer model".to_vec(),
            max_compute: 50000,
            caller: owner,
        };
        
        let result = runtime.execute_agent(request).await.unwrap();
        
        assert!(result.success);
        assert!(result.compute_units_used > 10000); // Should include model load cost
        assert!(!result.output_data.is_empty());
        
        // Verify output indicates simulated inference (since no model file exists)
        let output: serde_json::Value = serde_json::from_slice(&result.output_data).unwrap();
        assert_eq!(output["inference_type"], "simulated");
    }

    #[tokio::test]
    async fn test_execution_history() {
        let (runtime, _temp_dir) = create_test_runtime();
        
        let agent_id = vec![1u8; 32];
        let owner = vec![3u8; 32];
        let config = create_test_config();
        
        runtime.deploy_agent(agent_id.clone(), vec![2u8; 32], owner.clone(), config).unwrap();
        
        // Execute multiple times
        for i in 0..5 {
            let request = ExecutionRequest {
                agent_id: agent_id.clone(),
                input_data: vec![i; 10],
                max_compute: 50000,
                caller: owner.clone(),
            };
            runtime.execute_agent(request).await.unwrap();
        }
        
        let history = runtime.get_execution_history(&agent_id, 10);
        assert_eq!(history.len(), 5);
    }

    #[test]
    fn test_model_storage() {
        let (runtime, _temp_dir) = create_test_runtime();
        
        // Store a dummy model
        let model_bytes = b"dummy model content";
        let result = runtime.store_model(model_bytes);
        assert!(result.is_ok());
        
        let model_hash = result.unwrap();
        assert_eq!(model_hash.len(), 32);
    }
}
