mod embedding;
mod extractor;
mod pipeline;
mod search;
mod storage;
mod types;

pub use embedding::{EmbeddingError, EmbeddingModel, EmbeddingModelConfig};
pub use extractor::{parse_rollout, ParseError};
pub use pipeline::{
    process_rollout_dir, process_rollout_file, update_rollout_dir, PipelineError, UpdateStats,
};
pub use search::{search_with_text, search_with_vector, SearchError, SearchParams, SearchResult};
pub use storage::{RolloutFingerprint, Storage, StorageError};
pub use types::*;
