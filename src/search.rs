use bytemuck::cast_slice;
use rusqlite::types::Value as SqlValue;
use thiserror::Error;

use crate::embedding::{EmbeddingError, EmbeddingModel};
use crate::storage::Storage;

/// Parameters describing the metadata filters and limits applied to a search.
pub struct SearchParams<'a> {
    pub meta_equals: Vec<(&'a str, &'a str)>,
    pub conversation_ids: Vec<&'a str>,
    pub limit: usize,
    pub prefetch: Option<usize>,
}

impl<'a> SearchParams<'a> {
    /// Create a new parameter set with a desired result limit.
    pub fn new(limit: usize) -> Self {
        Self {
            meta_equals: Vec::new(),
            conversation_ids: Vec::new(),
            limit,
            prefetch: None,
        }
    }
}

impl<'a> Default for SearchParams<'a> {
    fn default() -> Self {
        SearchParams::new(10)
    }
}

/// Result row returned by a semantic search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub conversation_id: String,
    pub turn_index: usize,
    pub score: f32,
    pub user_text: Option<String>,
    pub assistant_text: Option<String>,
}

/// Errors produced while executing a search.
#[derive(Debug, Error)]
pub enum SearchError {
    #[error("sql error: {0}")]
    Sql(#[from] rusqlite::Error),
    #[error("invalid metadata filter key '{0}'")]
    InvalidMetaKey(String),
    #[error("embedding error: {0}")]
    Embedding(EmbeddingError),
}

/// Perform a semantic search by first generating an embedding for `text`.
pub fn search_with_text(
    storage: &Storage,
    embedder: &EmbeddingModel,
    text: &str,
    params: &SearchParams<'_>,
) -> Result<Vec<SearchResult>, SearchError> {
    let query_vector = embedder.embed(text).map_err(SearchError::Embedding)?;
    search_with_vector(storage, &query_vector, params)
}

/// Perform a semantic search using a pre-computed query vector.
pub fn search_with_vector(
    storage: &Storage,
    query_vector: &[f32],
    params: &SearchParams<'_>,
) -> Result<Vec<SearchResult>, SearchError> {
    if query_vector.is_empty() || params.limit == 0 {
        return Ok(Vec::new());
    }

    let mut sql = String::from(
        "SELECT t.conversation_id, t.turn_index, t.user_text, t.assistant_text, t.embedding \
         FROM turns t \
         JOIN conversations c ON c.id = t.conversation_id \
         WHERE t.embedding IS NOT NULL",
    );
    let mut values: Vec<SqlValue> = Vec::new();

    if !params.conversation_ids.is_empty() {
        sql.push_str(" AND t.conversation_id IN (");
        for (idx, _) in params.conversation_ids.iter().enumerate() {
            if idx > 0 {
                sql.push_str(", ");
            }
            sql.push('?');
        }
        sql.push(')');
        for id in &params.conversation_ids {
            values.push(SqlValue::from((*id).to_string()));
        }
    }

    for (key, value) in &params.meta_equals {
        ensure_valid_meta_key(key)?;
        sql.push_str(" AND json_extract(c.meta_json, '$.");
        sql.push_str(key);
        sql.push_str("') = ?");
        values.push(SqlValue::from((*value).to_string()));
    }

    let prefetch = params
        .prefetch
        .unwrap_or_else(|| params.limit.saturating_mul(8).max(params.limit));
    sql.push_str(" LIMIT ?");
    values.push(SqlValue::from(prefetch as i64));

    let conn = storage.connection();
    let mut stmt = conn.prepare(&sql)?;
    let params_refs: Vec<&dyn rusqlite::ToSql> =
        values.iter().map(|v| v as &dyn rusqlite::ToSql).collect();
    let mut rows = stmt.query(params_refs.as_slice())?;

    let query_norm = l2_norm(query_vector);
    if query_norm == 0.0 {
        return Ok(Vec::new());
    }

    let mut results: Vec<SearchResult> = Vec::new();

    while let Some(row) = rows.next()? {
        let conversation_id: String = row.get(0)?;
        let turn_index: i64 = row.get(1)?;
        if turn_index < 0 {
            continue;
        }
        let user_text: Option<String> = row.get(2)?;
        let assistant_text: Option<String> = row.get(3)?;
        let embedding_blob: Vec<u8> = row.get(4)?;
        if embedding_blob.is_empty() || embedding_blob.len() % std::mem::size_of::<f32>() != 0 {
            continue;
        }
        let embedding: Vec<f32> = cast_slice::<u8, f32>(&embedding_blob).to_vec();
        if embedding.len() != query_vector.len() {
            continue;
        }
        let score = cosine_similarity(query_vector, query_norm, &embedding);
        if !score.is_finite() {
            continue;
        }
        results.push(SearchResult {
            conversation_id,
            turn_index: turn_index as usize,
            score,
            user_text,
            assistant_text,
        });
    }

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if results.len() > params.limit {
        results.truncate(params.limit);
    }
    Ok(results)
}

fn cosine_similarity(query: &[f32], query_norm: f32, candidate: &[f32]) -> f32 {
    let candidate_norm = l2_norm(candidate);
    if candidate_norm == 0.0 {
        return 0.0;
    }
    let dot = query
        .iter()
        .zip(candidate.iter())
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum::<f64>();
    (dot / ((query_norm as f64) * (candidate_norm as f64))) as f32
}

fn l2_norm(vector: &[f32]) -> f32 {
    vector
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt() as f32
}

fn ensure_valid_meta_key(key: &str) -> Result<(), SearchError> {
    if key.is_empty() {
        return Err(SearchError::InvalidMetaKey(key.to_string()));
    }
    for segment in key.split('.') {
        if segment.is_empty()
            || !segment
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(SearchError::InvalidMetaKey(key.to_string()));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{RolloutFingerprint, Storage};
    use crate::types::{ConversationRecord, TurnRecord, TurnResult, TurnTelemetry};
    use serde_json::json;

    fn insert_turn_with_embedding(
        storage: &Storage,
        conversation_id: &str,
        assistant_text: &str,
        embedding: &[f32],
    ) {
        let turn = TurnRecord {
            index: 0,
            started_at: None,
            context: None,
            user_inputs: Vec::new(),
            result: TurnResult {
                assistant_messages: vec![assistant_text.to_string()],
                ..TurnResult::default()
            },
            actions: Vec::new(),
            telemetry: TurnTelemetry::default(),
        };
        storage
            .insert_turn(conversation_id, &turn, Some(embedding))
            .unwrap();
    }

    #[test]
    fn filters_and_ranks_results() {
        let storage = Storage::open_in_memory().unwrap();

        let mut record_alpha = ConversationRecord::default();
        record_alpha.session_meta = Some(json!({"id":"alpha","project":"alpha"}));
        let alpha_id = storage
            .upsert_conversation("alpha.jsonl", &record_alpha, &RolloutFingerprint::default())
            .unwrap();
        insert_turn_with_embedding(&storage, &alpha_id, "alpha result", &[1.0, 0.0]);

        let mut record_beta = ConversationRecord::default();
        record_beta.session_meta = Some(json!({"id":"beta","project":"beta"}));
        let beta_id = storage
            .upsert_conversation("beta.jsonl", &record_beta, &RolloutFingerprint::default())
            .unwrap();
        insert_turn_with_embedding(&storage, &beta_id, "beta result", &[0.0, 1.0]);

        let mut params = SearchParams::new(5);
        params.meta_equals.push(("project", "alpha"));

        let results = search_with_vector(&storage, &[1.0, 0.0], &params).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].conversation_id, "alpha");
        assert!(results[0]
            .assistant_text
            .as_ref()
            .unwrap()
            .contains("alpha"));

        let results = search_with_vector(&storage, &[0.0, 1.0], &SearchParams::new(5)).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].conversation_id, "beta");
    }

    #[test]
    fn rejects_bad_meta_keys() {
        let storage = Storage::open_in_memory().unwrap();
        let params = SearchParams {
            meta_equals: vec![("project'; DROP TABLE --", "alpha")],
            conversation_ids: Vec::new(),
            limit: 5,
            prefetch: None,
        };
        let err = search_with_vector(&storage, &[1.0], &params).unwrap_err();
        assert!(matches!(err, SearchError::InvalidMetaKey(_)));
    }
}
