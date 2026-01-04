use anyhow::{Result, Context};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde::{Serialize, Deserialize};
use tracing::{info, warn, debug};

use crate::utils::{current_timestamp, hash_data, to_base58};

/// AI Agent runtime environment
pub struct AIRuntime {
    agents: Arc<RwLock<HashMap<Vec<u8>, AIAgentState>>>,
    execution_cache: Arc<RwLock<HashMap<Vec<u8>, ExecutionCache>>>,
    max_concurrent_executions: usize,
    max_execution_time_ms: u64,
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
        Self {
            agents: Arc::new(RwLock::new(HashMap::new())),
            execution_cache: Arc::new(RwLock::new(HashMap::new())),
            max_concurrent_executions,
            max_execution_time_ms,
        }
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
        if let Some(cached) = self.get_cached_execution(&input_hash) {
            info!("Using cached execution result");
            return Ok(ExecutionResult {
                execution_id: self.generate_execution_id(&request.agent_id, &input_hash),
                output_data: cached.output,
                compute_units_used: cached.compute_used,
                logs: vec!["Using cached result".to_string()],
                success: true,
                error: None,
            });
        }

        let start_time = std::time::Instant::now();
        let mut logs = Vec::new();
        
        logs.push(format!("Starting execution for agent: {}", to_base58(&request.agent_id)));
        logs.push(format!("Max compute: {}", request.max_compute));

        let (output, compute_used) = self.simulate_ai_execution(&agent, &request.input_data, request.max_compute).await?;
        
        let execution_time = start_time.elapsed().as_millis() as u64;
        logs.push(format!("Execution completed in {}ms", execution_time));
        logs.push(format!("Compute units used: {}", compute_used));

        {
            let mut agents = self.agents.write().unwrap();
            if let Some(agent_state) = agents.get_mut(&request.agent_id) {
                agent_state.execution_count += 1;
                agent_state.total_compute_used += compute_used;
            }
        }

        self.cache_execution(input_hash.clone(), output.clone(), compute_used);

        let execution_id = self.generate_execution_id(&request.agent_id, &input_hash);

        Ok(ExecutionResult {
            execution_id,
            output_data: output,
            compute_units_used: compute_used,
            logs,
            success: true,
            error: None,
        })
    }

    async fn simulate_ai_execution(
        &self,
        agent: &AIAgentState,
        input_data: &[u8],
        max_compute: u64,
    ) -> Result<(Vec<u8>, u64)> {
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        
        let output = format!("AI output for input: {:?}", input_data.len()).into_bytes();
        let compute_used = (input_data.len() as u64 * 100).min(max_compute);
        
        Ok((output, compute_used))
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

    pub fn get_stats(&self) -> RuntimeStats {
        let agents = self.agents.read().unwrap();
        
        let total_agents = agents.len();
        let active_agents = agents.values().filter(|a| a.status == AgentStatus::Active).count();
        let total_executions: u64 = agents.values().map(|a| a.execution_count).sum();
        let total_compute: u64 = agents.values().map(|a| a.total_compute_used).sum();
        
        let cache = self.execution_cache.read().unwrap();
        
        RuntimeStats {
            total_agents,
            active_agents,
            total_executions,
            total_compute_used: total_compute,
            cache_size: cache.len(),
        }
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
        let mut cache = self.execution_cache.write().unwrap();
        
        if cache.len() >= 1000 {
            let oldest_keys: Vec<_> = cache.iter()
                .take(100)
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeStats {
    pub total_agents: usize,
    pub active_agents: usize,
    pub total_executions: u64,
    pub total_compute_used: u64,
    pub cache_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> AgentConfig {
        AgentConfig {
            model_type: "transformer".to_string(),
            parameters: HashMap::new(),
            max_execution_time: 30000,
            allowed_operations: vec!["inference".to_string()],
            compute_budget: 100000,
        }
    }

    #[test]
    fn test_deploy_agent() {
        let runtime = AIRuntime::new(10, 30000);
        
        let agent_id = vec![1u8; 32];
        let model_hash = vec![2u8; 32];
        let owner = vec![3u8; 32];
        let config = create_test_config();
        
        let result = runtime.deploy_agent(agent_id.clone(), model_hash, owner, config);
        assert!(result.is_ok());
        
        let info = runtime.get_agent_info(&agent_id);
        assert!(info.is_some());
    }

    #[test]
    fn test_duplicate_deployment() {
        let runtime = AIRuntime::new(10, 30000);
        
        let agent_id = vec![1u8; 32];
        let config = create_test_config();
        
        runtime.deploy_agent(agent_id.clone(), vec![2u8; 32], vec![3u8; 32], config.clone()).unwrap();
        
        let result = runtime.deploy_agent(agent_id, vec![2u8; 32], vec![3u8; 32], config);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_execute_agent() {
        let runtime = AIRuntime::new(10, 30000);
        
        let agent_id = vec![1u8; 32];
        let owner = vec![3u8; 32];
        let config = create_test_config();
        
        runtime.deploy_agent(agent_id.clone(), vec![2u8; 32], owner.clone(), config).unwrap();
        
        let request = ExecutionRequest {
            agent_id: agent_id.clone(),
            input_data: vec![1, 2, 3, 4],
            max_compute: 50000,
            caller: owner,
        };
        
        let result = runtime.execute_agent(request).await;
        assert!(result.is_ok());
        
        let execution = result.unwrap();
        assert!(execution.success);
        assert!(execution.compute_units_used > 0);
    }

    #[tokio::test]
    async fn test_execution_cache() {
        let runtime = AIRuntime::new(10, 30000);
        
        let agent_id = vec![1u8; 32];
        let owner = vec![3u8; 32];
        let config = create_test_config();
        
        runtime.deploy_agent(agent_id.clone(), vec![2u8; 32], owner.clone(), config).unwrap();
        
        let request = ExecutionRequest {
            agent_id: agent_id.clone(),
            input_data: vec![1, 2, 3, 4],
            max_compute: 50000,
            caller: owner.clone(),
        };
        
        let result1 = runtime.execute_agent(request.clone()).await.unwrap();
        let result2 = runtime.execute_agent(request).await.unwrap();
        
        assert_eq!(result1.output_data, result2.output_data);
    }

    #[test]
    fn test_pause_resume_agent() {
        let runtime = AIRuntime::new(10, 30000);
        
        let agent_id = vec![1u8; 32];
        let owner = vec![3u8; 32];
        let config = create_test_config();
        
        runtime.deploy_agent(agent_id.clone(), vec![2u8; 32], owner.clone(), config).unwrap();
        
        runtime.pause_agent(&agent_id, &owner).unwrap();
        let info = runtime.get_agent_info(&agent_id).unwrap();
        assert_eq!(info.status, AgentStatus::Paused);
        
        runtime.resume_agent(&agent_id, &owner).unwrap();
        let info = runtime.get_agent_info(&agent_id).unwrap();
        assert_eq!(info.status, AgentStatus::Active);
    }

    #[test]
    fn test_unauthorized_update() {
        let runtime = AIRuntime::new(10, 30000);
        
        let agent_id = vec![1u8; 32];
        let owner = vec![3u8; 32];
        let not_owner = vec![4u8; 32];
        let config = create_test_config();
        
        runtime.deploy_agent(agent_id.clone(), vec![2u8; 32], owner, config.clone()).unwrap();
        
        let result = runtime.update_agent(&agent_id, &not_owner, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_list_agents_by_owner() {
        let runtime = AIRuntime::new(10, 30000);
        
        let owner = vec![3u8; 32];
        let config = create_test_config();
        
        runtime.deploy_agent(vec![1u8; 32], vec![2u8; 32], owner.clone(), config.clone()).unwrap();
        runtime.deploy_agent(vec![2u8; 32], vec![2u8; 32], owner.clone(), config.clone()).unwrap();
        
        let agents = runtime.list_agents_by_owner(&owner);
        assert_eq!(agents.len(), 2);
    }
}
