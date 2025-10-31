use std::fs;
use std::path::{Path, PathBuf};

use conv_memory::{
    process_rollout_dir, search_with_vector, update_rollout_dir, ConversationStats, SearchParams,
    Storage,
};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use tempfile::{tempdir, TempDir};

const SAMPLE_EMBED_DIM: usize = 64;
const ROLLOUTS_FOR_IMPORT: usize = 64;
const TURNS_PER_ROLLOUT: usize = 6;

fn bench_import_rollouts(c: &mut Criterion) {
    let rollouts = generate_rollouts(ROLLOUTS_FOR_IMPORT, TURNS_PER_ROLLOUT);
    c.bench_function("import_rollout_dir", |b| {
        b.iter_batched(
            || setup_rollout_dir(&rollouts),
            |(dir, db_path)| {
                let storage = Storage::open(&db_path).expect("open storage");
                process_rollout_dir(dir.path(), &storage, None).expect("import rollouts");
                black_box(storage);
            },
            BatchSize::LargeInput,
        )
    });
}

fn bench_update_rollouts(c: &mut Criterion) {
    let base_rollouts = generate_rollouts(ROLLOUTS_FOR_IMPORT, TURNS_PER_ROLLOUT);
    let mut updated_rollouts = base_rollouts.clone();
    if let Some(first) = updated_rollouts.first_mut() {
        *first = tweak_rollout(first, "assistant updated");
    }

    c.bench_function("update_rollout_dir", |b| {
        b.iter_batched(
            || {
                let (dir, db_path) = setup_rollout_dir(&base_rollouts);
                {
                    let storage = Storage::open(&db_path).expect("open storage");
                    process_rollout_dir(dir.path(), &storage, None).expect("initial import");
                }
                let first_path = discover_rollout_paths(dir.path())
                    .expect("discover rollouts")
                    .into_iter()
                    .next()
                    .expect("at least one rollout");
                fs::write(&first_path, updated_rollouts[0].as_bytes()).expect("rewrite rollout");
                (dir, db_path)
            },
            |(dir, db_path)| {
                let storage = Storage::open(&db_path).expect("open storage");
                let stats =
                    update_rollout_dir(dir.path(), &storage, None).expect("update rollouts");
                black_box(stats);
            },
            BatchSize::LargeInput,
        )
    });
}

fn bench_search_queries(c: &mut Criterion) {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("conv.sqlite");
    let storage = Storage::open(&db_path).expect("open storage");

    seed_search_data(&storage, 128, 4);

    let query = generate_embedding(SAMPLE_EMBED_DIM, 9);
    let mut params = SearchParams::new(10);
    params.prefetch = Some(64);

    c.bench_function("search_vector", |b| {
        b.iter(|| {
            let results = search_with_vector(&storage, black_box(&query), &params).unwrap();
            black_box(results);
        })
    });
}

fn generate_rollouts(count: usize, turns: usize) -> Vec<String> {
    (0..count).map(|idx| render_rollout(idx, turns)).collect()
}

fn render_rollout(index: usize, turns: usize) -> String {
    let mut lines = Vec::new();
    let base = 1_700_000_000_u64 + (index as u64 * 20);
    lines.push(format!(
        "{{\"timestamp\":\"{}\",\"type\":\"session_meta\",\"payload\":{{\"id\":\"bench-{:04}\"}}}}",
        iso_timestamp(base),
        index
    ));
    for turn in 0..turns {
        let user_ts = base + (turn as u64) * 2 + 1;
        let assistant_ts = user_ts + 1;
        lines.push(format!(
            "{{\"timestamp\":\"{}\",\"type\":\"response_item\",\"payload\":{{\"type\":\"message\",\"role\":\"user\",\"content\":[{{\"type\":\"input_text\",\"text\":\"hello {}\"}}]}}}}",
            iso_timestamp(user_ts), turn
        ));
        lines.push(format!(
            "{{\"timestamp\":\"{}\",\"type\":\"response_item\",\"payload\":{{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{{\"type\":\"output_text\",\"text\":\"response {} {}\"}}]}}}}",
            iso_timestamp(assistant_ts), index, turn
        ));
    }
    lines.join("\n")
}

fn tweak_rollout(original: &str, new_word: &str) -> String {
    original
        .lines()
        .map(|line| {
            if line.contains("response") {
                line.replace("response", new_word)
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn iso_timestamp(epoch_seconds: u64) -> String {
    use time::{Duration, OffsetDateTime};
    let epoch = OffsetDateTime::from_unix_timestamp(0).unwrap();
    let ts = epoch + Duration::seconds(epoch_seconds as i64);
    ts.format(&time::format_description::well_known::Rfc3339)
        .unwrap()
}

fn setup_rollout_dir(rollouts: &[String]) -> (TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    for (idx, contents) in rollouts.iter().enumerate() {
        let nested = dir.path().join(format!("2025/10/bench-{:04}", idx));
        fs::create_dir_all(&nested).expect("mkdir nested");
        let file_path = nested.join(format!(
            "rollout-2025-10-{:02}T00-00-{:02}-bench.jsonl",
            (idx % 30) + 1,
            idx % 60
        ));
        fs::write(&file_path, contents).expect("write rollout");
    }
    let db_path = dir.path().join("storage.sqlite");
    (dir, db_path)
}

fn discover_rollout_paths(dir: &Path) -> Result<Vec<PathBuf>, std::io::Error> {
    let mut paths = Vec::new();
    for entry in walkdir::WalkDir::new(dir) {
        let entry = entry?;
        if entry.file_type().is_file() {
            let name = entry.file_name().to_string_lossy();
            if name.starts_with("rollout-") && name.ends_with(".jsonl") {
                paths.push(entry.into_path());
            }
        }
    }
    paths.sort();
    Ok(paths)
}

fn seed_search_data(storage: &Storage, conversations: usize, turns_per_conversation: usize) {
    use conv_memory::{
        ConversationRecord, RolloutFingerprint, TurnRecord, TurnResult, TurnTelemetry,
        UserInputRecord,
    };
    use serde_json::json;

    for idx in 0..conversations {
        let record = ConversationRecord {
            session_meta: Some(json!({
                "id": format!("conv-{idx:04}"),
                "project": if idx % 2 == 0 { "alpha" } else { "beta" }
            })),
            ..ConversationRecord::default()
        };
        let fingerprint = RolloutFingerprint {
            modified_at: None,
            size_bytes: Some(256),
            sha256: Some(format!("{:032x}", idx)),
        };
        let mut stats = ConversationStats::default();
        stats.turn_count = turns_per_conversation as i64;
        stats.questions = vec!["Benchmark".to_string()];
        stats.search_blob = format!("benchmark conv-{idx:04}");
        stats.cwd = Some(format!("/tmp/bench/{idx:04}"));
        let conversation_id = storage
            .upsert_conversation(
                format!("bench-conv-{idx:04}.jsonl"),
                &record,
                &fingerprint,
                &stats,
                None,
            )
            .expect("insert conversation");

        for turn_idx in 0..turns_per_conversation {
            let turn = TurnRecord {
                index: turn_idx,
                started_at: None,
                context: None,
                user_inputs: vec![UserInputRecord {
                    raw: json!({"type":"message","role":"user","content":"Benchmark"}),
                    text: Some("Benchmark".into()),
                    images: Vec::new(),
                }],
                result: TurnResult {
                    assistant_messages: vec![format!("Answer {idx:04}-{turn_idx:02}")],
                    ..TurnResult::default()
                },
                actions: Vec::new(),
                telemetry: TurnTelemetry::default(),
            };
            let embedding =
                generate_embedding(SAMPLE_EMBED_DIM, (idx as u64) << 16 | turn_idx as u64);
            storage
                .insert_turn(&conversation_id, &turn, Some(&embedding))
                .expect("insert turn");
        }
    }
}

fn generate_embedding(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(42 ^ seed);
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

criterion_group!(
    name = perf_benches;
    config = Criterion::default().sample_size(20);
    targets = bench_import_rollouts, bench_update_rollouts, bench_search_queries
);
criterion_main!(perf_benches);
