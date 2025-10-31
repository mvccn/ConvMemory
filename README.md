# ConvMemory

## The problem

Agentic coding sessions with Codex, Cursor, Claude Code, and similar tools generate a huge trail of artefacts: successful runs, code scan results, root-cause analyses, half-finished explorations, and the heuristics that made a solution work. Today most systems stash these insights manually in “add memory” files (AGENTS.md, CLAUDE.md, etc.), then trawl through raw rollouts or notebooks to rediscover them. Manual recall is tedious, inconsistent, and too expensive during an active incident. We need a systematic way to accumulate the skills and knowledge from every coding run so the next run starts smarter. ConvMemory is that system.

## ConvMemory

ConvMemory builds adapters (starting with Codex rollouts—more agent formats are planned), normalises every turn, vectorises the summaries, and persists everything to SQLite. The design goal is maximum import and query throughput on local hardware: Rust orchestrates the pipeline, GGUF models provide embeddings, and the resulting store stays small and portable.

- No external service is required.
- ConvMemory is fast and compact.

| Scenario                                     | Raw rollouts (`~/.codex/sessions`)                       | ConvMemory                                                                  |
| -------------------------------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------- |
| Vector similarity search                     | N/A                                                      | ~31 µs per query (with model load, 1-2s for model warm-up)                  |
| Text search across 1,617 rollouts            | ~1.32 s to traverse and parse JSONL with a Python script | ~0.77 s for a `SELECT … LIKE` over 45 k indexed turns                       |
| Full import (1,617 real rollouts)            | N/A                                                      | ~10ms/rollout                                                               |
| Incremental update after one rollout changes | Re-scan entire tree (~16.6 s)                            | ~7.3 ms via `update_rollout_dir` (Criterion, 1 s warm-up / 2 s measurement) |

_Benchmarks collected on the same Mac Pro used for Codex development; raw figures come from direct filesystem traversal scripts while ConvMemory timings use the included Criterion suite or SQLite queries._

It provides:

- A parser that normalises Codex `rollout-*.jsonl` sessions into per-turn records (user inputs, assistant output, tools used, telemetry, token counts).
- A storage layer backed by SQLite for conversation metadata, turn transcripts, telemetry, and optional vector embeddings.
- Optional on-device embeddings (via `llama_cpp` + GGUF models) so downstream workflows can perform semantic search.
- A CLI importer (`conv-memory-import`) that can ingest individual files or whole directories of rollouts in batch.

| Scenario                                     | Raw rollouts (`~/.codex/sessions`)                       | ConvMemory                                                                  |
| -------------------------------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------- |
| Vector similarity search                     | N/A                                                      | ~31 µs per query (with model load, 1-2s for model warm-up)                  |
| Text search across 1,617 rollouts            | ~1.32 s to traverse and parse JSONL with a Python script | ~0.77 s for a `SELECT … LIKE` over 45 k indexed turns                       |
| Full import (1,617 real rollouts)            | N/A                                                      | ~10ms/rollout                                                               |
| Incremental update after one rollout changes | Re-scan entire tree (~16.6 s)                            | ~7.3 ms via `update_rollout_dir` (Criterion, 1 s warm-up / 2 s measurement) |

_Benchmarks collected on the same Mac Pro used for Codex development; raw figures come from direct filesystem traversal scripts while ConvMemory timings use the included Criterion suite or SQLite queries._

It provides:

- A parser that normalises Codex `rollout-*.jsonl` sessions into per-turn records (user inputs, assistant output, tools used, telemetry, token counts).
- A storage layer backed by SQLite for conversation metadata, turn transcripts, telemetry, and optional vector embeddings.
- Optional on-device embeddings (via `llama_cpp` + GGUF models) so downstream workflows can perform semantic search.
- A CLI importer (`conv-memory-import`) that can ingest individual files or whole directories of rollouts in batch.

## Getting started

1. Install the Rust toolchain (stable) if you have not already: <https://rustup.rs>.
2. Enter the crate directory:

   ```bash
   cd ConvMemory
   ```

3. Run the test suite:

   ```bash
   cargo test
   ```

   To validate the embedding runtime as well, enable the feature:

   ```bash
   cargo test --features embedding-runtime
   ```

### add embedding model

Embedding support uses [`llama_cpp`](https://crates.io/crates/llama_cpp) with Metal acceleration on macOS. Download a GGUF model (the project currently targets the Nomic embed model):

```bash
mkdir models
cd models
curl -L -o models/nomic-embed-text-v1.5.Q4_K_M.gguf \
  https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf
```

You can customise GPU offload and CPU threading with CLI flags or by editing the `EmbeddingModelConfig`.

## CLI usage

Build and run the importer with Cargo:

```bash
cargo run --bin conv-memory-import -- --help
```

Ingest the default Codex session directory (relative to the repository root) into a SQLite file:

```bash
cargo run --bin conv-memory-import -- \
  ../sessions \
  --database conv-memory.sqlite
```

If you keep rollouts under `~/.codex/sessions`, point the importer there instead:

```bash
cargo run --bin conv-memory-import -- \
  ~/.codex/sessions \
  --database conv-memory.sqlite
```

Include embeddings by providing the GGUF path and any runtime tuning you need:

```bash
cargo run --features embedding-runtime --bin conv-memory-import -- \
  ~/.codex/sessions \
  --database conv-memory.sqlite \
  --embed-model models/nomic-embed-text-v1.5.Q4_K_M.gguf \
  --embed-gpu-layers 1 \
  --embed-threads 6
```

## Database schema

The SQLite schema is created automatically on first run:

- `conversations` stores rollout-level metadata (path, timestamps, duration, token usage, embedding dimension, and raw metadata JSON).
- `turns` stores per-turn transcripts, telemetry snapshots, and optional embedding vectors.

The schema is designed so you can introduce secondary indexes or vector-search extensions (e.g. `sqlite-vec`) later without changing the importer.

## Incremental ingestion

When you need to keep a long-running knowledge base fresh, call the library API instead of the CLI:

```rust
use conv_memory::{process_rollout_dir, update_rollout_dir, Storage};

let storage = Storage::open("conv-memory.sqlite")?;
// One-off full backfill.
process_rollout_dir("/Users/grad/.codex/sessions", &storage, None)?;

// Later, only new or changed rollouts are reprocessed.
let stats = update_rollout_dir("/Users/grad/.codex/sessions", &storage, None)?;
println!("updated {} rollouts, skipped {}", stats.processed, stats.skipped);
```

Each conversation row records the source file’s modified time, size, and SHA-256 hash so `update_rollout_dir` can skip unchanged rollouts while still refreshing files that grew new turns.

## Semantic search helpers

ConvMemory exposes an in-process vector search that filters by session metadata before scoring embeddings:

```rust
use conv_memory::{search_with_text, SearchParams};

let mut params = SearchParams::new(10);
params.meta_equals.push(("project", "codex"));

let results = search_with_text(&storage, &embedder, "how did we fix websocket auth?", &params)?;
for hit in results {
    println!("{}#{} score={:.3}", hit.conversation_id, hit.turn_index, hit.score);
}
```

- `SearchParams` lets you constrain results by metadata (`meta_equals`) or conversation IDs before vectors are loaded.
- Use `search_with_vector` if you already have an embedding and want to avoid recomputing it.
- Only turns with stored embeddings participate; run imports with an embedder to populate the vectors column.

## Performance benchmarks

Criterion benches cover end-to-end ingestion, incremental updates, and vector search latency with synthetic data:

```bash
cargo bench --bench performance -- --warm-up-time 1 --measurement-time 2
```

Results are written to `target/criterion/` and include HTML reports (requires `open target/criterion/index.html`). Adjust the constants in `benches/performance.rs` to scale the dataset size up or down for stress testing your environment. Benchmarks (median): import of 64 rollouts completes in ≈38 ms; incremental updates that touch a single rollout finish in ≈7.3 ms; vector search queries return in ≈31 µs (Criterion, 1 s warm-up, 2 s measurement).

## Development tips

- Use `cargo fmt` before committing code changes.
- When adding embedding-related functionality, gate it behind the `embedding-runtime` feature so the crate still builds without the model or Metal support.
- The pipeline helpers in `src/pipeline.rs` are reusable from other binaries or services if you outgrow the CLI.

## Embedding ConvMemory in Rust

Programmatic usage mirrors the CLI: open a database, ingest rollouts, then refresh and query as needed.

```rust
use conv_memory::{
    process_rollout_dir, search_with_text, update_rollout_dir, EmbeddingModel,
    EmbeddingModelConfig, SearchParams, Storage,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let storage = Storage::open("conv-memory.sqlite")?;

    // One-off full backfill.
    process_rollout_dir("/Users/grad/.codex/sessions", &storage, None)?;

    // Incremental refresh that only revisits changed rollouts.
    let stats = update_rollout_dir("/Users/grad/.codex/sessions", &storage, None)?;
    println!("updated {} rollouts, skipped {}", stats.processed, stats.skipped);

    #[cfg(feature = "embedding-runtime")]
    {
        let embedder = EmbeddingModel::load(EmbeddingModelConfig::new(
            "models/nomic-embed-text-v1.5.Q4_K_M.gguf",
        ))?;
        let mut params = SearchParams::new(5);
        params.meta_equals.push(("project", "codex"));

        let hits = search_with_text(
            &storage,
            &embedder,
            "how did we fix websocket auth?",
            &params,
        )?;

        for hit in hits {
            println!("{}#{} {:.3}", hit.conversation_id, hit.turn_index, hit.score);
        }
    }

    Ok(())
}
```

The library API returns structured results (`TurnRecord`, telemetry snapshots, embeddings) so downstream services can enrich or persist them in other systems without reparsing raw JSONL.
