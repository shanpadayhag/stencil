//! `stencil train` — rebuild the suggestive models from the logged records.
//!
//! Thin orchestration over [`crate::ml`]: read a task's JSONL log, map each record to features +
//! labels (the T63 mappers), fit both heads with a CV-tuned threshold ([`crate::ml::train`]), and
//! atomically swap the resulting artifact into the task's data dir. No flags trains both models;
//! `--styling` / `--censor` scope to one. Training only ever happens on this explicit command.

use std::path::Path;

use anyhow::{Context, Result};

use crate::cli::TrainArgs;
use crate::commands::read_jsonl;
use crate::learn::{self, DecisionRecord, Model, StylingRecord};
use crate::ml::artifact::{self, TaskModel};
use crate::ml::features::{censor, styling};
use crate::ml::logreg::FitOptions;
use crate::ml::train::{self, TrainOutcome, TrainingData};

/// Run the `train` subcommand.
pub fn run(args: TrainArgs) -> Result<()> {
    // No flags → both models; otherwise just the named one(s).
    let both = !args.styling && !args.censor;
    let trained_at = learn::now_epoch_secs().to_string();
    let opts = FitOptions::default();

    if args.styling || both {
        train_styling(&args, &trained_at, &opts)?;
    }
    if args.censor || both {
        train_censor(&args, &trained_at, &opts)?;
    }
    Ok(())
}

/// Train the styling model from `styling.jsonl`.
fn train_styling(args: &TrainArgs, trained_at: &str, opts: &FitOptions) -> Result<()> {
    let dir = learn::model_dir(
        Model::Styling,
        args.data_dir.as_deref(),
        args.styling_dir.as_deref(),
    )?;
    let records: Vec<StylingRecord> = read_jsonl(&dir.join("styling.jsonl"))?;
    let data = TrainingData {
        features: records.iter().map(styling::styling_features).collect(),
        verdict_labels: records
            .iter()
            .map(|record| u8::from(styling::is_weird(record)))
            .collect(),
        reasons: records
            .iter()
            .map(|record| styling::reason(record).map(str::to_string))
            .collect(),
    };
    finish("styling", data, trained_at, opts, &dir.join("model.json"))
}

/// Train the censor model from `decisions.jsonl`.
fn train_censor(args: &TrainArgs, trained_at: &str, opts: &FitOptions) -> Result<()> {
    let dir = learn::model_dir(
        Model::Censor,
        args.data_dir.as_deref(),
        args.censor_dir.as_deref(),
    )?;
    let records: Vec<DecisionRecord> = read_jsonl(&dir.join("decisions.jsonl"))?;
    let data = TrainingData {
        features: records.iter().map(censor::censor_features).collect(),
        verdict_labels: records
            .iter()
            .map(|record| u8::from(censor::is_confirm(record)))
            .collect(),
        reasons: records
            .iter()
            .map(|record| censor::reason(record).map(str::to_string))
            .collect(),
    };
    finish("censor", data, trained_at, opts, &dir.join("model.json"))
}

/// Train, atomically swap on success, and print the per-model summary.
fn finish(
    label: &str,
    data: TrainingData,
    trained_at: &str,
    opts: &FitOptions,
    model_path: &Path,
) -> Result<()> {
    match train::train(&data, trained_at.to_string(), opts) {
        TrainOutcome::Trained(model) => {
            print_trained(label, &model);
            artifact::save_atomic(model_path, &model)
                .with_context(|| format!("failed to write the {label} model"))?;
        }
        TrainOutcome::NotEnoughData {
            n_records,
            positives,
        } => {
            println!(
                "{label}: not enough data to train ({n_records} record(s), {positives} positive(s)); \
                 no model written."
            );
        }
    }
    Ok(())
}

/// Print the trained-model summary: record count + held-out (CV) balanced accuracy + reason classes.
fn print_trained(label: &str, model: &TaskModel) {
    let reason_classes = model
        .reason
        .as_ref()
        .map_or(0, |reason| reason.classes.len());
    let reason_note = if reason_classes >= 2 {
        format!("{reason_classes} reason classes")
    } else {
        "no reason head".to_string()
    };
    println!(
        "{label}: trained on {} record(s) — CV balanced accuracy {:.1}% ({reason_note}).",
        model.n_records,
        model.cv_balanced_accuracy * 100.0,
    );
}
