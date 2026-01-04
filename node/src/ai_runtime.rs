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

    /// Store a model by its bytes and return the hash
    pub fn store_model(&self, model_bytes: &[u8]) -> Result<Vec<u8>> {
        let model_hash = hash_data(model_bytes);
        let hash_str = to_base58(&model_hash);
        
        // Create model storage directory if it doesn't exist
        std::fs::create_dir_all(&self.model_storage_path)
            .context("Failed to create model storage directory")?;
        
        let bytes_path = format!("{}/{}.bin", self.model_storage_path, hash_str);
        std::fs::write(&bytes_path, model_bytes)
            .context("Failed to store model bytes")?;
        
        info!("Stored model with hash: {}", hash_str);
        Ok(model_hash)
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
        let (output, execution_cost) = self.execute_with_real_model(
            &model, 
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
    async fn execute_with_real_model(
        &self,
        model: &OnnxModel,
        input_data: &[u8],
        config: &AgentConfig,
        remaining_compute: u64,
    ) -> Result<(Vec<u8>, u64)> {
        let start_time = std::time::Instant::now();
        
        // Parse input based on model type
        let output_data = match config.model_type.as_str() {
            "transformer" | "embeddings" => {
                // Text input - use infer_text
                let input_text = String::from_utf8_lossy(input_data);
                let max_length = config.parameters.get("maxTokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(512) as usize;
                
                let output = model.infer_text(&input_text, max_length)?;
                
                serde_json::json!({
                    "model": config.model_type,
                    "input_length": input_text.len(),
                    "output_shape": [output.len()],
                    "output": output,
                    "timestamp": current_timestamp(),
                    "inference_type": "real"
                }).to_string().into_bytes()
            }
            "cnn" | "rnn" | _ => {
                // Binary/tensor input - convert to float array
                let input_floats: Vec<f32> = input_data.iter()
                    .map(|&b| b as f32 / 255.0) // Normalize to [0, 1]
                    .collect();
                
                let input_shape = IxDyn(&[1, input_floats.len()]);
                let input_array = ArrayD::from_shape_vec(input_shape, input_floats)
                    .context("Failed to create input tensor")?;
                
                let outputs = model.infer(input_array)?;
                
                let output_vecs: Vec<Vec<f32>> = outputs.iter()
                    .map(|arr| arr.iter().cloned().collect())
                    .collect();
                
                serde_json::json!({
                    "model": config.model_type,
                    "input_size": input_data.len(),
                    "output_count": outputs.len(),
                    "outputs": output_vecs,
                    "timestamp": current_timestamp(),
                    "inference_type": "real"
                }).to_string().into_bytes()
            }
        };
        
        let execution_time_ms = start_time.elapsed().as_millis() as u64;
        
        // Compute cost based on actual execution time
        // 1 compute unit per microsecond
        let compute_cost = (start_time.elapsed().as_micros() as u64)
            .min(remaining_compute);
        
        info!("Real model inference completed in {}ms, compute cost: {}", execution_time_ms, compute_cost);
        
        Ok((output_data, compute_cost))
    }

    /// Execute transformer model (simulated fallback)
    async fn execute_transformer_simulated(
        &self,
        input_data: &[u8],
        config: &AgentConfig,
    ) -> Result<(Vec<u8>, u64)> {
        // Parse input as text
        let input_text = String::from_utf8_lossy(input_data);
        
        // Get parameters
        let max_tokens = config.parameters.get("maxTokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(512);
        
        let temperature = config.parameters.get("temperature")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.7);

        // Estimate token count (rough: 1 token ≈ 4 chars)
        let input_tokens = (input_text.len() / 4) as u64;
        
        // Compute cost: base + per-token
        let base_cost = 1_000u64;
        let per_token_cost = 100u64;
        let compute_cost = base_cost + (input_tokens * per_token_cost) + (max_tokens * per_token_cost);

        // Generate simulated output
        let output = serde_json::json!({
            "model": "transformer",
            "input_length": input_text.len(),
            "input_tokens": input_tokens,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "result": format!("Processed: {} chars", input_text.len()),
            "timestamp": current_timestamp(),
            "inference_type": "simulated"
        });

        Ok((output.to_string().into_bytes(), compute_cost))
    }

    /// Execute CNN model (simulated fallback)
    async fn execute_cnn_simulated(
        &self,
        input_data: &[u8],
        config: &AgentConfig,
    ) -> Result<(Vec<u8>, u64)> {
        // CNN typically processes images/structured data
        let input_size = input_data.len() as u64;
        
        // Compute cost based on input size
        let base_cost = 5_000u64;
        let processing_cost = input_size * 2; // 2 compute units per byte
        let compute_cost = base_cost + processing_cost;

        let output = serde_json::json!({
            "model": "cnn",
            "input_size": input_size,
            "features_extracted": input_size / 100, // Simplified
            "timestamp": current_timestamp(),
            "inference_type": "simulated"
        });

        Ok((output.to_string().into_bytes(), compute_cost))
    }

    /// Execute RNN model (simulated fallback)
    async fn execute_rnn_simulated(
        &self,
        input_data: &[u8],
        _config: &AgentConfig,
    ) -> Result<(Vec<u8>, u64)> {
        let sequence_length = input_data.len() / 4; // Assume 4 bytes per element
        
        let base_cost = 7_000u64;
        let per_step_cost = 75u64;
        let compute_cost = base_cost + (sequence_length as u64 * per_step_cost);

        let output = serde_json::json!({
            "model": "rnn",
            "sequence_length": sequence_length,
            "timestamp": current_timestamp(),
            "inference_type": "simulated"
        });

        Ok((output.to_string().into_bytes(), compute_cost))
    }

    /// Execute embeddings model (simulated fallback)
    async fn execute_embeddings_simulated(
        &self,
        input_data: &[u8],
        config: &AgentConfig,
    ) -> Result<(Vec<u8>, u64)> {
        let input_text = String::from_utf8_lossy(input_data);
        let input_tokens = (input_text.len() / 4) as u64;
        
        let embedding_dim = config.parameters.get("embeddingDim")
            .and_then(|v| v.as_u64())
            .unwrap_or(768);

        let base_cost = 500u64;
        let per_token_cost = 50u64;
        let dim_multiplier = (embedding_dim / 100).max(1);
        let compute_cost = base_cost + (input_tokens * per_token_cost * dim_multiplier);

        // Generate deterministic pseudo-embeddings based on input hash
        let input_hash = hash_data(input_data);
        let seed = u64::from_le_bytes(input_hash[0..8].try_into().unwrap_or([0u8; 8]));
        let embeddings: Vec<f32> = (0..embedding_dim)
            .map(|i| {
                let val = ((seed.wrapping_add(i)) % 1000) as f32 / 1000.0;
                (val * 2.0) - 1.0 // Normalize to [-1, 1]
            })
            .collect();

        let output = serde_json::json!({
            "model": "embeddings",
            "input_tokens": input_tokens,
            "embedding_dim": embedding_dim,
            "embeddings": embeddings,
            "timestamp": current_timestamp(),
            "inference_type": "simulated"
        });

        Ok((output.to_string().into_bytes(), compute_cost))
    }

    /// Execute generic model (simulated fallback)
    async fn execute_generic_simulated(
        &self,
        input_data: &[u8],
        _config: &AgentConfig,
    ) -> Result<(Vec<u8>, u64)> {
        let input_size = input_data.len() as u64;
        
        let base_cost = 8_000u64;
        let processing_cost = input_size * 80;
        let compute_cost = base_cost + processing_cost;

        let output = serde_json::json!({
            "model": "generic",
            "input_size": input_size,
            "processed": true,
            "timestamp": current_timestamp(),
            "inference_type": "simulated"
        });

        Ok((output.to_string().into_bytes(), compute_cost))
    }

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
