use std::fs::{self, Metadata};
use std::io::Cursor;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use thiserror::Error;
use time::OffsetDateTime;
use walkdir::WalkDir;

use crate::embedding::{EmbeddingError, EmbeddingModel};
use crate::extractor::{parse_rollout, ParseError};
use crate::storage::{RolloutFingerprint, Storage, StorageError};
use crate::types::TurnRecord;

/// Errors surfaced when processing and persisting rollout files.
#[derive(Error, Debug)]
pub enum PipelineError {
    #[error("parse error: {0}")]
    Parse(#[from] ParseError),
    #[error("embedding error: {0}")]
    Embedding(#[from] EmbeddingError),
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("walkdir error: {0}")]
    WalkDir(#[from] walkdir::Error),
}

/// Process a single rollout file, generating embeddings (when an embedder is provided) and
/// storing results in SQLite.
pub fn process_rollout_file(
    rollout_path: impl AsRef<Path>,
    storage: &Storage,
    embedder: Option<&EmbeddingModel>,
) -> Result<(), PipelineError> {
    let rollout_path = rollout_path.as_ref();
    let (bytes, fingerprint) = load_rollout_data(rollout_path, None)?;
    ingest_rollout_bytes(rollout_path, &bytes, &fingerprint, storage, embedder)
}

/// Process every rollout file under `dir`, returning the number of files that were ingested.
pub fn process_rollout_dir(
    dir: impl AsRef<Path>,
    storage: &Storage,
    embedder: Option<&EmbeddingModel>,
) -> Result<usize, PipelineError> {
    let rollouts = discover_rollouts(dir.as_ref())?;
    let mut processed = 0usize;
    for path in rollouts {
        process_rollout_file(&path, storage, embedder)?;
        processed += 1;
    }
    Ok(processed)
}

/// Incrementally process rollout files under `dir`, skipping those whose metadata has not changed.
pub fn update_rollout_dir(
    dir: impl AsRef<Path>,
    storage: &Storage,
    embedder: Option<&EmbeddingModel>,
) -> Result<UpdateStats, PipelineError> {
    let rollouts = discover_rollouts(dir.as_ref())?;
    let mut stats = UpdateStats::default();

    for path in rollouts {
        let metadata = fs::metadata(&path)?;
        let (modified_at, size_bytes) = file_metadata(&metadata);

        if let Some(existing) = storage.get_rollout_fingerprint(&path)? {
            if fingerprint_matches(&existing, modified_at, size_bytes) {
                stats.skipped += 1;
                continue;
            }
        }

        let (bytes, fingerprint) = load_rollout_data(&path, Some(&metadata))?;
        ingest_rollout_bytes(&path, &bytes, &fingerprint, storage, embedder)?;
        stats.processed += 1;
    }

    Ok(stats)
}

/// Summary of incremental update work.
#[derive(Debug, Default)]
pub struct UpdateStats {
    pub processed: usize,
    pub skipped: usize,
}

fn discover_rollouts(dir: &Path) -> Result<Vec<PathBuf>, PipelineError> {
    let mut rollouts: Vec<PathBuf> = Vec::new();
    for entry in WalkDir::new(dir) {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy();
        if name.starts_with("rollout-") && name.ends_with(".jsonl") {
            rollouts.push(entry.into_path());
        }
    }
    rollouts.sort();
    Ok(rollouts)
}

fn load_rollout_data(
    path: &Path,
    metadata: Option<&Metadata>,
) -> Result<(Vec<u8>, RolloutFingerprint), PipelineError> {
    let owned_meta;
    let meta = match metadata {
        Some(m) => m,
        None => {
            owned_meta = fs::metadata(path)?;
            &owned_meta
        }
    };

    let bytes = fs::read(path)?;
    let (modified_at, size_bytes) = file_metadata(meta);
    let sha256 = Some(format!("{:x}", Sha256::digest(&bytes)));

    Ok((
        bytes,
        RolloutFingerprint {
            modified_at,
            size_bytes,
            sha256,
        },
    ))
}

fn ingest_rollout_bytes(
    rollout_path: &Path,
    bytes: &[u8],
    fingerprint: &RolloutFingerprint,
    storage: &Storage,
    embedder: Option<&EmbeddingModel>,
) -> Result<(), PipelineError> {
    let cursor = Cursor::new(bytes);
    let record = parse_rollout(cursor)?;

    let conversation_id = storage.upsert_conversation(rollout_path, &record, fingerprint)?;

    let embeddings = if let Some(embedder) = embedder {
        let summaries: Vec<String> = record.turns.iter().map(render_turn_summary).collect();
        let vectors = embedder.embed_batch(&summaries)?;
        if vectors.len() != record.turns.len() {
            return Err(PipelineError::Embedding(EmbeddingError::MissingOutput));
        }
        Some(vectors)
    } else {
        None
    };

    for (idx, turn) in record.turns.iter().enumerate() {
        let embedding_slice = embeddings.as_ref().map(|vecs| vecs[idx].as_slice());
        storage.insert_turn(&conversation_id, turn, embedding_slice)?;
    }

    Ok(())
}

fn fingerprint_matches(
    existing: &RolloutFingerprint,
    modified_at: Option<OffsetDateTime>,
    size_bytes: Option<u64>,
) -> bool {
    match (existing.modified_at, modified_at) {
        (Some(stored), Some(current)) if stored == current => {}
        _ => return false,
    }

    match (existing.size_bytes, size_bytes) {
        (Some(stored), Some(current)) if stored == current => {}
        _ => return false,
    }

    true
}

fn file_metadata(meta: &Metadata) -> (Option<OffsetDateTime>, Option<u64>) {
    let modified_at = meta.modified().ok().map(OffsetDateTime::from);
    let size_bytes = Some(meta.len());
    (modified_at, size_bytes)
}

fn render_turn_summary(turn: &TurnRecord) -> String {
    let mut sections = Vec::new();

    if !turn.user_inputs.is_empty() {
        let mut rendered_inputs = Vec::new();
        for (idx, input) in turn.user_inputs.iter().enumerate() {
            let mut fragment = String::new();
            if let Some(text) = &input.text {
                fragment.push_str(text);
            }
            if !input.images.is_empty() {
                if !fragment.is_empty() {
                    fragment.push('\n');
                }
                fragment.push_str(&format!("[{} image(s)]", input.images.len()));
            }
            if !fragment.is_empty() {
                rendered_inputs.push(format!("#{} {}", idx + 1, fragment.trim()));
            }
        }
        if !rendered_inputs.is_empty() {
            sections.push(format!("User:\n{}", rendered_inputs.join("\n\n")));
        }
    }

    let mut result_texts = Vec::new();
    if !turn.result.assistant_messages.is_empty() {
        result_texts.push(turn.result.assistant_messages.join("\n\n"));
    }
    if let Some(fallback) = &turn.result.fallback {
        result_texts.push(format!(
            "[fallback {:?}] {}",
            fallback.source, fallback.text
        ));
    }
    if !result_texts.is_empty() {
        sections.push(format!("Assistant:\n{}", result_texts.join("\n\n")));
    }

    if !turn.actions.is_empty() {
        let mut action_summaries = Vec::new();
        for action in &turn.actions {
            let summary = match &action.kind {
                crate::types::ActionKind::FunctionCall { name } => format!(
                    "function_call {}",
                    name.clone().unwrap_or_else(|| "(unknown)".into())
                ),
                crate::types::ActionKind::CustomToolCall { name } => format!(
                    "custom_tool {}",
                    name.clone().unwrap_or_else(|| "(unknown)".into())
                ),
                crate::types::ActionKind::LocalShellExec {
                    command, workdir, ..
                } => {
                    let joined = command.join(" ");
                    if let Some(dir) = workdir {
                        format!("shell `{}` (cwd: {})", joined, dir)
                    } else {
                        format!("shell `{}`", joined)
                    }
                }
                crate::types::ActionKind::WebSearch { query } => format!(
                    "web_search {}",
                    query.clone().unwrap_or_else(|| "(query missing)".into())
                ),
                crate::types::ActionKind::Other { kind } => {
                    format!("{}", kind.clone().unwrap_or_else(|| "other".into()))
                }
            };

            let status = action
                .status
                .status_text
                .clone()
                .or(action.status.local_status.clone());

            let mut rendered = format!("- {}", summary);
            if let Some(call_id) = &action.call_id {
                rendered.push_str(&format!(" (call_id={})", call_id));
            }
            if let Some(status) = status {
                rendered.push_str(&format!(" [status: {}]", status));
            }
            if let Some(output) = &action.output {
                if let Some(content) = &output.content {
                    let snippet = content.trim();
                    if !snippet.is_empty() {
                        let shortened = snippet.chars().take(200).collect::<String>();
                        rendered.push_str(&format!(" -> {}", shortened));
                    }
                }
            }
            action_summaries.push(rendered);
        }
        if !action_summaries.is_empty() {
            sections.push(format!("Actions:\n{}", action_summaries.join("\n")));
        }
    }

    if sections.is_empty() {
        "No transcript recorded for this turn.".to_string()
    } else {
        sections.join("\n\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::Storage;
    use std::io::Write;
    use std::time::Duration;
    use tempfile::{tempdir, NamedTempFile};

    fn sample_rollout() -> String {
        sample_rollout_with_assistant("hi there")
    }

    fn sample_rollout_with_assistant(text: &str) -> String {
        format!(
            r#"
{{"timestamp":"2025-01-01T00:00:00.000Z","type":"session_meta","payload":{{"id":"urn:uuid:test","cwd":"/tmp"}}}}
{{"timestamp":"2025-01-01T00:00:01.000Z","type":"response_item","payload":{{"type":"message","role":"user","content":[{{"type":"input_text","text":"hello"}}]}}}}
{{"timestamp":"2025-01-01T00:00:02.000Z","type":"response_item","payload":{{"type":"message","role":"assistant","content":[{{"type":"output_text","text":"{text}"}}]}}}}
"#,
            text = text
        )
    }

    #[test]
    fn pipeline_stores_rollout_without_embeddings() {
        let mut tmp = NamedTempFile::new().unwrap();
        let contents = sample_rollout();
        tmp.write_all(contents.as_bytes()).unwrap();
        tmp.flush().unwrap();

        let storage = Storage::open_in_memory().unwrap();
        process_rollout_file(tmp.path(), &storage, None).unwrap();

        let count: i64 = storage
            .connection()
            .query_row("SELECT COUNT(*) FROM turns", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn pipeline_processes_directory() {
        let dir = tempdir().unwrap();
        let nested = dir.path().join("2025/10/01");
        std::fs::create_dir_all(&nested).unwrap();
        let file_path = nested.join("rollout-2025-10-01T00-00-00-abc.jsonl");
        std::fs::write(&file_path, sample_rollout()).unwrap();

        let storage = Storage::open_in_memory().unwrap();
        let processed = process_rollout_dir(dir.path(), &storage, None).unwrap();
        assert_eq!(processed, 1);

        let count: i64 = storage
            .connection()
            .query_row("SELECT COUNT(*) FROM turns", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn update_dir_skips_unchanged_and_refreshes_modified_files() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("rollout-2025-10-01T00-00-00-abc.jsonl");
        std::fs::write(&file_path, sample_rollout()).unwrap();

        let storage = Storage::open_in_memory().unwrap();
        let processed = process_rollout_dir(dir.path(), &storage, None).unwrap();
        assert_eq!(processed, 1);

        let stats = update_rollout_dir(dir.path(), &storage, None).unwrap();
        assert_eq!(stats.processed, 0);
        assert_eq!(stats.skipped, 1);

        std::thread::sleep(Duration::from_millis(1100));
        std::fs::write(
            &file_path,
            sample_rollout_with_assistant("updated response"),
        )
        .unwrap();

        let stats = update_rollout_dir(dir.path(), &storage, None).unwrap();
        assert_eq!(stats.processed, 1);
        assert_eq!(stats.skipped, 0);

        let assistant: String = storage
            .connection()
            .query_row(
                "SELECT assistant_text FROM turns WHERE conversation_id = (SELECT id FROM conversations LIMIT 1)",
                [],
                |row| row.get::<_, Option<String>>(0),
            )
            .unwrap()
            .unwrap();
        assert!(assistant.contains("updated response"));
    }
}
