use std::path::{Path, PathBuf};

use thiserror::Error;

#[cfg(feature = "embedding-runtime")]
use llama_cpp::{EmbeddingsParams, LlamaModel, LlamaParams};
#[cfg(feature = "embedding-runtime")]
use num_cpus;

/// Configuration parameters for the on-device embedding model.
#[derive(Debug, Clone)]
pub struct EmbeddingModelConfig {
    /// Path to the GGUF model on disk.
    pub model_path: PathBuf,
    /// Number of transformer layers to offload to the GPU. `None` keeps the library default.
    pub gpu_layers: Option<u32>,
    /// Number of CPU threads to use during embedding. Defaults to `num_cpus::get_physical() - 1`.
    pub threads: Option<u32>,
    /// Number of CPU threads to use for batch operations. Defaults to the same value as `threads`.
    pub threads_batch: Option<u32>,
}

impl EmbeddingModelConfig {
    /// Create a new configuration from a model path.
    pub fn new(model_path: impl AsRef<Path>) -> Self {
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            gpu_layers: None,
            threads: None,
            threads_batch: None,
        }
    }
}

/// Errors produced by the embedding runtime.
#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[cfg(feature = "embedding-runtime")]
    #[error("failed to load model: {0}")]
    Load(#[from] llama_cpp::LlamaLoadError),
    #[cfg(feature = "embedding-runtime")]
    #[error("embedding inference failed: {0}")]
    Inference(#[from] llama_cpp::LlamaContextError),
    #[error("embedding output missing")]
    MissingOutput,
    #[error("embedding runtime not available in this build; recompile with the `embedding-runtime` feature")]
    Unavailable,
}

#[cfg(feature = "embedding-runtime")]
pub struct EmbeddingModel {
    model: LlamaModel,
    threads: u32,
    threads_batch: u32,
}

#[cfg(feature = "embedding-runtime")]
impl EmbeddingModel {
    /// Load the GGUF model and prepare it for embedding inference.
    pub fn load(config: EmbeddingModelConfig) -> Result<Self, EmbeddingError> {
        let mut params = LlamaParams::default();
        if let Some(layers) = config.gpu_layers {
            params.n_gpu_layers = layers;
        }
        params.use_mmap = true;
        params.use_mlock = false;

        let model = LlamaModel::load_from_file(config.model_path, params)?;
        let threads = config
            .threads
            .unwrap_or_else(|| (num_cpus::get_physical().saturating_sub(1)).max(1) as u32);
        let threads_batch = config.threads_batch.unwrap_or(threads);

        Ok(Self {
            model,
            threads,
            threads_batch,
        })
    }

    fn embedding_params(&self) -> EmbeddingsParams {
        EmbeddingsParams {
            n_threads: self.threads,
            n_threads_batch: self.threads_batch,
        }
    }

    /// Generate an embedding vector for the provided text.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let embeddings = self.model.embeddings(&[text], self.embedding_params())?;
        embeddings
            .into_iter()
            .next()
            .ok_or(EmbeddingError::MissingOutput)
    }

    /// Generate embeddings for a batch of inputs.
    pub fn embed_batch(&self, inputs: &[impl AsRef<str>]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let owned: Vec<String> = inputs.iter().map(|s| s.as_ref().to_string()).collect();
        let refs: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();
        let embeddings = self.model.embeddings(&refs, self.embedding_params())?;
        Ok(embeddings)
    }

    /// The dimensionality of vectors produced by this model.
    pub fn embedding_dim(&self) -> usize {
        self.model.embed_len()
    }
}

#[cfg(not(feature = "embedding-runtime"))]
pub struct EmbeddingModel;

#[cfg(not(feature = "embedding-runtime"))]
impl EmbeddingModel {
    pub fn load(_config: EmbeddingModelConfig) -> Result<Self, EmbeddingError> {
        Err(EmbeddingError::Unavailable)
    }

    pub fn embed(&self, _text: &str) -> Result<Vec<f32>, EmbeddingError> {
        Err(EmbeddingError::Unavailable)
    }

    pub fn embed_batch(
        &self,
        _inputs: &[impl AsRef<str>],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        Err(EmbeddingError::Unavailable)
    }

    pub fn embedding_dim(&self) -> usize {
        0
    }
}

#[cfg(all(test, feature = "embedding-runtime"))]
mod tests {
    use super::*;
    use std::env;

    fn locate_model() -> Option<PathBuf> {
        if let Ok(path) = env::var("CONVMEMORY_EMBED_MODEL") {
            return Some(PathBuf::from(path));
        }
        let default = Path::new("./models/nomic-embed-text-v1.5.Q4_K_M.gguf");
        if default.exists() {
            Some(default.to_path_buf())
        } else {
            None
        }
    }

    #[test]
    fn embeds_example_text() {
        let Some(model_path) = locate_model() else {
            eprintln!("embedding test skipped: model file not found");
            return;
        };

        let model = EmbeddingModel::load(EmbeddingModelConfig {
            model_path,
            gpu_layers: Some(1),
            threads: Some(4),
            threads_batch: Some(4),
        })
        .expect("failed to load embedding model");

        let vector = model
            .embed("Hello from ConvMemory!")
            .expect("embedding request failed");
        assert!(!vector.is_empty());
    }
}
