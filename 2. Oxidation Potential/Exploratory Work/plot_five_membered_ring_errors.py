from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_DIR / "chemprop_model_output"
DEFAULT_COLUMNS = 4


def latest_model_dir(model_root: Path) -> Path:
    model_dirs = [path for path in model_root.iterdir() if path.is_dir() and path.name.startswith("model")]
    if not model_dirs:
        raise FileNotFoundError(f"No model directories found in {model_root}")
    return max(model_dirs, key=lambda path: path.stat().st_mtime)


def has_five_membered_ring(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return any(len(ring) == 5 for ring in mol.GetRingInfo().AtomRings())


def split_sort_key(split_name: str) -> int:
    order = {"train": 0, "val": 1, "test": 2}
    return order.get(str(split_name).lower(), 99)


def build_subset(predictions_path: Path) -> pd.DataFrame:
    df = pd.read_csv(predictions_path)
    required_cols = {"SMILES", "OP_true", "OP_pred", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {predictions_path}: {sorted(missing)}")

    subset = df[df["SMILES"].map(has_five_membered_ring)].copy()
    subset["residual"] = subset["OP_true"] - subset["OP_pred"]
    subset["abs_error"] = subset["residual"].abs()
    subset["five_membered_ring_count"] = subset["SMILES"].map(
        lambda smiles: sum(len(ring) == 5 for ring in Chem.MolFromSmiles(smiles).GetRingInfo().AtomRings())
    )
    subset["_split_order"] = subset["split"].map(split_sort_key)
    subset = subset.sort_values(["_split_order", "abs_error"], ascending=[True, False]).reset_index(drop=True)
    subset["row_id"] = subset.index + 1
    return subset.drop(columns="_split_order")


def plot_subset(subset: pd.DataFrame, output_png: Path, columns: int = DEFAULT_COLUMNS) -> None:
    if subset.empty:
        raise ValueError("No five-membered-ring molecules found to plot.")

    rows = math.ceil(len(subset) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 4.6 * rows), dpi=220)
    if not isinstance(axes, (list, tuple)):
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    else:
        axes = list(axes)

    for ax, row in zip(axes, subset.itertuples(index=False), strict=False):
        mol = Chem.MolFromSmiles(row.SMILES)
        image = Draw.MolToImage(mol, size=(500, 380))
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(
            f"#{row.row_id} | {row.split}\n"
            f"abs err = {row.abs_error:.3f} V\n"
            f"resid = {row.residual:.3f} V",
            fontsize=9,
            fontweight="bold",
        )
        ax.text(
            0.5,
            -0.08,
            f"true {row.OP_true:.3f} | pred {row.OP_pred:.3f}\n"
            f"5-membered rings: {row.five_membered_ring_count}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )

    for ax in axes[len(subset) :]:
        ax.axis("off")

    split_counts = subset["split"].value_counts().to_dict()
    fig.suptitle(
        "Five-Membered-Ring Molecules in Latest OP Predictions\n"
        f"n={len(subset)} | split counts={split_counts}",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot five-membered-ring molecules with prediction errors.")
    parser.add_argument("--model-dir", type=Path, default=None, help="Specific model directory. Defaults to latest.")
    parser.add_argument("--columns", type=int, default=DEFAULT_COLUMNS, help="Number of columns in the grid figure.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir if args.model_dir is not None else latest_model_dir(MODEL_ROOT)
    predictions_path = model_dir / "OP_predictions_complete.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_path}")

    subset = build_subset(predictions_path)
    output_csv = model_dir / "OP_five_membered_ring_error_table.csv"
    output_png = model_dir / "OP_five_membered_ring_error_grid.png"
    subset.to_csv(output_csv, index=False)
    plot_subset(subset, output_png=output_png, columns=args.columns)

    print(f"Model directory: {model_dir}")
    print(f"Predictions source: {predictions_path}")
    print(f"Five-membered-ring rows: {len(subset)}")
    print(f"Table saved to: {output_csv}")
    print(f"Figure saved to: {output_png}")
    print("\nSplit counts:")
    print(subset["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
