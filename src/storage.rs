use std::path::Path;

use bytemuck::cast_slice;
use rusqlite::{params, Connection, OpenFlags};
use serde_json::Value;
use thiserror::Error;
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use crate::types::{ConversationRecord, FallbackSource, TokenUsageBreakdown, TurnRecord};

/// Errors surfaced by the storage layer.
#[derive(Error, Debug)]
pub enum StorageError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Simple SQLite-backed persistence for conversations and turn embeddings.
pub struct Storage {
    conn: Connection,
}

/// Fingerprint describing the rollout file that produced a conversation.
#[derive(Debug, Clone, Default)]
pub struct RolloutFingerprint {
    pub modified_at: Option<OffsetDateTime>,
    pub size_bytes: Option<u64>,
    pub sha256: Option<String>,
}

impl Storage {
    /// Open (or create) the database at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
        )?;
        setup_schema(&conn)?;
        Ok(Self { conn })
    }

    /// Create an in-memory database. Handy for tests.
    #[cfg(test)]
    pub fn open_in_memory() -> Result<Self, StorageError> {
        let conn = Connection::open_in_memory()?;
        setup_schema(&conn)?;
        Ok(Self { conn })
    }

    /// Insert or update conversation metadata and return the conversation id we stored under.
    pub fn upsert_conversation(
        &self,
        rollout_path: impl AsRef<Path>,
        record: &ConversationRecord,
        fingerprint: &RolloutFingerprint,
    ) -> Result<String, StorageError> {
        let rollout_path = rollout_path.as_ref();
        let conversation_id = extract_conversation_id(record, rollout_path);

        let meta_json = record
            .session_meta
            .as_ref()
            .map(|v| serde_json::to_string(v))
            .transpose()?;

        let started_at = record.started_at.map(|ts| ts.to_string());
        let ended_at = record.ended_at.map(|ts| ts.to_string());
        let duration_seconds = record.duration_seconds.map(|d| d as i64);

        let token_total = best_breakdown(record)
            .and_then(|b| b.total_tokens)
            .map(|v| v as i64);
        let token_input = best_breakdown(record)
            .and_then(|b| b.input_tokens)
            .map(|v| v as i64);
        let token_cached = best_breakdown(record)
            .and_then(|b| b.cached_input_tokens)
            .map(|v| v as i64);
        let token_output = best_breakdown(record)
            .and_then(|b| b.output_tokens)
            .map(|v| v as i64);
        let token_reasoning = best_breakdown(record)
            .and_then(|b| b.reasoning_output_tokens)
            .map(|v| v as i64);
        let model_ctx = record.token_usage.model_context_window.map(|v| v as i64);
        let modified_at = fingerprint
            .modified_at
            .and_then(|ts| ts.format(&Rfc3339).ok());
        let size_bytes = fingerprint.size_bytes.map(|v| v as i64);
        let sha256 = fingerprint.sha256.clone();

        self.conn.execute(
            r#"
            INSERT INTO conversations
            (id, rollout_path, started_at, ended_at, duration_seconds, token_input, token_cached,
             token_output, token_reasoning, token_total, token_model_context, meta_json,
             rollout_modified_at, rollout_size_bytes, rollout_hash)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)
            ON CONFLICT(id) DO UPDATE SET
                rollout_path = excluded.rollout_path,
                started_at = excluded.started_at,
                ended_at = excluded.ended_at,
                duration_seconds = excluded.duration_seconds,
                token_input = excluded.token_input,
                token_cached = excluded.token_cached,
                token_output = excluded.token_output,
                token_reasoning = excluded.token_reasoning,
                token_total = excluded.token_total,
                token_model_context = excluded.token_model_context,
                meta_json = excluded.meta_json,
                rollout_modified_at = excluded.rollout_modified_at,
                rollout_size_bytes = excluded.rollout_size_bytes,
                rollout_hash = excluded.rollout_hash
            "#,
            params![
                conversation_id,
                rollout_path.to_string_lossy(),
                started_at,
                ended_at,
                duration_seconds,
                token_input,
                token_cached,
                token_output,
                token_reasoning,
                token_total,
                model_ctx,
                meta_json,
                modified_at,
                size_bytes,
                sha256,
            ],
        )?;

        Ok(conversation_id)
    }

    /// Persist a turn and its embedding.
    pub fn insert_turn(
        &self,
        conversation_id: &str,
        turn: &TurnRecord,
        embedding: Option<&[f32]>,
    ) -> Result<(), StorageError> {
        let started_at = turn.started_at.map(|ts| ts.to_string());
        let user_text = join_user_inputs(turn);
        let assistant_text = join_assistant_messages(turn);
        let fallback_text = turn.result.fallback.as_ref().map(|f| format_fallback(f));
        let actions_json = serde_json::to_string(&turn.actions)?;
        let telemetry_json = serde_json::to_string(&turn.telemetry)?;

        let embedding_blob = embedding.map(|vec| cast_slice::<f32, u8>(vec).to_vec());

        self.conn.execute(
            r#"
            INSERT INTO turns
            (conversation_id, turn_index, started_at, user_text, assistant_text, fallback_text,
             actions_json, telemetry_json, embedding)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
            ON CONFLICT(conversation_id, turn_index) DO UPDATE SET
                started_at = excluded.started_at,
                user_text = excluded.user_text,
                assistant_text = excluded.assistant_text,
                fallback_text = excluded.fallback_text,
                actions_json = excluded.actions_json,
                telemetry_json = excluded.telemetry_json,
                embedding = excluded.embedding
            "#,
            params![
                conversation_id,
                turn.index as i64,
                started_at,
                user_text,
                assistant_text,
                fallback_text,
                actions_json,
                telemetry_json,
                embedding_blob,
            ],
        )?;

        if let Some(embedding) = embedding {
            let dim = embedding.len() as i64;
            self.conn.execute(
                "UPDATE conversations SET embedding_dim = ?1 WHERE id = ?2 AND (embedding_dim IS NULL OR embedding_dim = ?1)",
                params![dim, conversation_id],
            )?;
        }

        Ok(())
    }

    /// Expose raw connection for advanced queries.
    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    /// Fetch stored fingerprint information for a rollout path, if present.
    pub fn get_rollout_fingerprint(
        &self,
        rollout_path: impl AsRef<Path>,
    ) -> Result<Option<RolloutFingerprint>, StorageError> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT rollout_modified_at, rollout_size_bytes, rollout_hash
            FROM conversations
            WHERE rollout_path = ?1
            LIMIT 1
            "#,
        )?;
        let mut rows = stmt.query(params![rollout_path.as_ref().to_string_lossy()])?;
        if let Some(row) = rows.next()? {
            let modified_at: Option<String> = row.get(0)?;
            let size_bytes: Option<i64> = row.get(1)?;
            let sha256: Option<String> = row.get(2)?;
            let parsed_modified =
                modified_at.and_then(|ts| OffsetDateTime::parse(&ts, &Rfc3339).ok());
            Ok(Some(RolloutFingerprint {
                modified_at: parsed_modified,
                size_bytes: size_bytes.map(|v| v as u64),
                sha256,
            }))
        } else {
            Ok(None)
        }
    }
}

fn join_user_inputs(turn: &TurnRecord) -> Option<String> {
    let mut texts: Vec<String> = Vec::new();
    for input in &turn.user_inputs {
        if let Some(text) = &input.text {
            texts.push(text.clone());
        }
    }
    if texts.is_empty() {
        None
    } else {
        Some(texts.join("\n\n"))
    }
}

fn join_assistant_messages(turn: &TurnRecord) -> Option<String> {
    if turn.result.assistant_messages.is_empty() {
        None
    } else {
        Some(turn.result.assistant_messages.join("\n\n"))
    }
}

fn format_fallback(fallback: &crate::types::FallbackSummary) -> String {
    match fallback.source {
        FallbackSource::AssistantReasoning => format!("[reasoning] {}", fallback.text),
        FallbackSource::ToolOutput => format!("[tool] {}", fallback.text),
        FallbackSource::EventStream => format!("[event] {}", fallback.text),
    }
}

fn best_breakdown(record: &ConversationRecord) -> Option<&TokenUsageBreakdown> {
    record
        .token_usage
        .total
        .as_ref()
        .or(record.token_usage.last.as_ref())
}

fn extract_conversation_id(record: &ConversationRecord, fallback_path: &Path) -> String {
    let from_meta = record
        .session_meta
        .as_ref()
        .and_then(|meta| meta.get("id").and_then(Value::as_str))
        .or_else(|| {
            record
                .session_meta
                .as_ref()
                .and_then(|meta| meta.get("conversation_id").and_then(Value::as_str))
        });

    if let Some(id) = from_meta {
        id.to_string()
    } else {
        // Fall back to the rollout filename to keep results deterministic.
        fallback_path
            .file_stem()
            .map(|stem| stem.to_string_lossy().to_string())
            .unwrap_or_else(|| fallback_path.to_string_lossy().to_string())
    }
}

fn setup_schema(conn: &Connection) -> Result<(), StorageError> {
    conn.execute_batch(
        r#"
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            rollout_path TEXT NOT NULL,
            started_at TEXT,
            ended_at TEXT,
            duration_seconds INTEGER,
            token_input INTEGER,
            token_cached INTEGER,
            token_output INTEGER,
            token_reasoning INTEGER,
            token_total INTEGER,
            token_model_context INTEGER,
            embedding_dim INTEGER,
            meta_json TEXT,
            rollout_modified_at TEXT,
            rollout_size_bytes INTEGER,
            rollout_hash TEXT
        );

        CREATE TABLE IF NOT EXISTS turns (
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            turn_index INTEGER NOT NULL,
            started_at TEXT,
            user_text TEXT,
            assistant_text TEXT,
            fallback_text TEXT,
            actions_json TEXT,
            telemetry_json TEXT,
            embedding BLOB,
            PRIMARY KEY (conversation_id, turn_index)
        );

        CREATE INDEX IF NOT EXISTS idx_turns_conversation ON turns(conversation_id);
        "#,
    )?;
    ensure_column(conn, "conversations", "rollout_modified_at", "TEXT")?;
    ensure_column(conn, "conversations", "rollout_size_bytes", "INTEGER")?;
    ensure_column(conn, "conversations", "rollout_hash", "TEXT")?;
    Ok(())
}

fn ensure_column(
    conn: &Connection,
    table: &str,
    column: &str,
    ty: &str,
) -> Result<(), StorageError> {
    let mut stmt = conn.prepare(format!("PRAGMA table_info({table})").as_str())?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let name: String = row.get(1)?;
        if name == column {
            return Ok(());
        }
    }
    let sql = format!("ALTER TABLE {table} ADD COLUMN {column} {ty}");
    let _ = conn.execute(sql.as_str(), []);
    Ok(())
}
