use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueHint};
use conv_memory::{
    process_rollout_dir, process_rollout_file, EmbeddingModel, EmbeddingModelConfig, Storage,
};

/// Import Codex rollout transcripts into the ConvMemory SQLite store.
#[derive(Debug, Parser)]
#[command(
    name = "conv-memory-import",
    version,
    about = "Batch ingest Codex rollouts into the ConvMemory knowledge base"
)]
struct Cli {
    /// Path to a rollout file or directory tree (defaults to ./codex/sessions).
    #[arg(
        value_name = "SOURCE",
        default_value = "codex/sessions",
        value_hint = ValueHint::AnyPath
    )]
    source: PathBuf,

    /// SQLite database to create or update.
    #[arg(
        short,
        long,
        value_name = "DB",
        default_value = "conv-memory.sqlite",
        value_hint = ValueHint::FilePath
    )]
    database: PathBuf,

    /// Optional GGUF embedding model for vectorising turn summaries.
    #[arg(long, value_name = "MODEL", value_hint = ValueHint::FilePath)]
    embed_model: Option<PathBuf>,

    /// Transformer layers offloaded to the GPU (Metal).
    #[arg(long, value_name = "N")]
    embed_gpu_layers: Option<u32>,

    /// CPU threads to use for embedding inference.
    #[arg(long, value_name = "THREADS")]
    embed_threads: Option<u32>,

    /// CPU threads to use for embedding batches.
    #[arg(long, value_name = "THREADS")]
    embed_threads_batch: Option<u32>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    if cli.embed_model.is_none()
        && (cli.embed_gpu_layers.is_some()
            || cli.embed_threads.is_some()
            || cli.embed_threads_batch.is_some())
    {
        eprintln!("warning: embedding flags were set without --embed-model; they will be ignored");
    }

    let storage = Storage::open(&cli.database)?;

    let embedder = if let Some(model_path) = &cli.embed_model {
        let config = EmbeddingModelConfig {
            model_path: model_path.clone(),
            gpu_layers: cli.embed_gpu_layers,
            threads: cli.embed_threads,
            threads_batch: cli.embed_threads_batch,
        };
        Some(EmbeddingModel::load(config)?)
    } else {
        None
    };

    let metadata = fs::metadata(&cli.source).map_err(|err| {
        format!(
            "failed to read source {}: {err}",
            cli.source.to_string_lossy()
        )
    })?;

    let start = Instant::now();

    if metadata.is_file() {
        process_rollout_file(&cli.source, &storage, embedder.as_ref())?;
        println!(
            "Imported rollout {} in {:.2?}",
            cli.source.display(),
            start.elapsed()
        );
    } else if metadata.is_dir() {
        let count = process_rollout_dir(&cli.source, &storage, embedder.as_ref())?;
        println!(
            "Imported {count} rollout(s) from {} in {:.2?}",
            cli.source.display(),
            start.elapsed()
        );
    } else {
        return Err(format!(
            "source {} is neither a file nor a directory",
            cli.source.display()
        )
        .into());
    }

    Ok(())
}
