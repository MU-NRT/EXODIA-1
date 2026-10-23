"""Validate a Chemprop OP run and write the usual prediction artifacts.

Run from the repository root as ``python '2. Oxidation Potential/Scripts/finalize_op_model.py' 53``.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "Datasets" / "trainable_data" / "OP_chemprop_train_file.csv"
FEATURES = ROOT / "Datasets" / "trainable_data" / "OP_chemprop_feat_file.npz"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(model_number):
    model_dir = ROOT / "chemprop_model_output" / f"model{model_number}"
    preds_file = model_dir / "predictions" / "set_preds.csv"
    best_pt = model_dir / "model_0" / "best.pt"
    metrics_file = model_dir / "model_0" / "trainer_logs" / "version_0" / "metrics.csv"
    for path in (TRAIN, FEATURES, preds_file, best_pt, metrics_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    train = pd.read_csv(TRAIN)
    preds = pd.read_csv(preds_file)
    with np.load(FEATURES) as archive:
        descriptor_rows = len(archive["arr_0"])
    if len(train) != len(preds) or len(train) != descriptor_rows:
        raise ValueError("Model data, predictions and descriptors have different row counts")
    if not train.SMILES.equals(preds.SMILES) or not train.split.equals(preds.split):
        raise ValueError("Predictions do not align with the model input rows")
    if not set(train.split) == {"train", "val", "test"}:
        raise ValueError("Expected train, val and test splits")

    complete = pd.DataFrame({
        "SMILES": train.SMILES,
        "OP_true": train.OP,
        "OP_pred": preds.OP,
        "split": train.split,
    })
    if not np.isfinite(complete[["OP_true", "OP_pred"]].to_numpy()).all():
        raise ValueError("Missing or nonfinite target/prediction")
    complete_file = model_dir / "OP_predictions_complete.csv"
    complete.to_csv(complete_file, index=False)

    metric_rows = []
    for split, part in complete.groupby("split"):
        residual = part.OP_pred - part.OP_true
        metric_rows.append({
            "split": split,
            "n": len(part),
            "MAE_V": float(residual.abs().mean()),
            "RMSE_V": float(np.sqrt((residual**2).mean())),
        })
    summary_file = model_dir / "split_metrics_summary.csv"
    pd.DataFrame(metric_rows).to_csv(summary_file, index=False)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "property_code": "OP",
        "model_dir": str(model_dir),
        "train_file": str(TRAIN),
        "train_file_sha256": sha256(TRAIN),
        "descriptor_file": str(FEATURES),
        "descriptor_file_sha256": sha256(FEATURES),
        "best_pt": str(best_pt),
        "predictions_file": str(preds_file),
        "merged_output_file": str(complete_file),
        "split_metrics_file": str(summary_file),
        "Gao_DFT_rows_removed": 330,
        "Gao_removal_audit": str(ROOT / "Datasets" / "trainable_data" / "Gao_computational_rows_removed.csv"),
        "epochs": 100,
        "batch_size": 8,
        "depth": 3,
        "dropout": 0.025,
        "ffn_hidden_dim": 50,
        "ffn_num_layers": 2,
        "seed": 11,
        "use_features": True,
        "use_transfer": False,
        "train_cmd": [
            "chemprop", "train", "--data-path", str(TRAIN),
            "--task-type", "regression", "--output-dir", str(model_dir),
            "--batch-size", "8", "--epochs", "100", "--depth", "3",
            "--dropout", "0.025", "--ffn-hidden-dim", "50",
            "--ffn-num-layers", "2", "--pytorch-seed", "11",
            "--splits-column", "split", "--descriptors-path", str(FEATURES),
        ],
        "predict_cmd": [
            "chemprop", "predict", "--test-path", str(TRAIN),
            "--model-paths", str(model_dir), "--preds-path", str(preds_file),
            "--descriptors-path", str(FEATURES),
        ],
    }
    (model_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame(metric_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("Checkpoint:", best_pt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_number", type=int)
    main(parser.parse_args().model_number)
