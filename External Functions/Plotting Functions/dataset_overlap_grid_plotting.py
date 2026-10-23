from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from rdkit import Chem


_PARITY_RED = "#c65a45"
_PASTEL_BLUE = "#a8cce9a6"
_WHITE = "#ffffff"
_DEFAULT_GRID_SIZE = 200
_DEFAULT_SAMPLE_SIZE = 40000
_DEFAULT_RANDOM_SEED = 42


@dataclass(frozen=True)
class PropertyDataset:
    code: str
    label: str
    filename: str


PROPERTY_DATASETS: tuple[PropertyDataset, ...] = (
    PropertyDataset(code="VP", label="Vapor Pressure", filename="VP_chemprop_train_file.csv"),
    PropertyDataset(code="OP", label="Oxidation Potential", filename="OP_chemprop_train_file.csv"),
    PropertyDataset(code="pKa", label="pKa", filename="pka_chemprop_train_file.csv"),
    PropertyDataset(code="MP", label="Melting Point", filename="MP_chemprop_train_file.csv"),
    PropertyDataset(code="BP", label="Boiling Point", filename="BP_chemprop_train_file.csv"),
)


def canonicalize_smiles(smiles: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _resolve_smiles_column(df: pd.DataFrame) -> str:
    for candidate in ("SMILES", "smiles"):
        if candidate in df.columns:
            return candidate
    raise ValueError("No SMILES column found. Expected 'SMILES' or 'smiles'.")


def load_canonical_smiles_set(csv_path: Path) -> set[str]:
    df = pd.read_csv(csv_path)
    smiles_col = _resolve_smiles_column(df)
    canonical_smiles = df[smiles_col].map(canonicalize_smiles).dropna().unique().tolist()
    return set(canonical_smiles)


def load_property_smiles_sets(data_dir: Path) -> dict[str, set[str]]:
    property_sets: dict[str, set[str]] = {}
    for dataset in PROPERTY_DATASETS:
        csv_path = data_dir / dataset.filename
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")
        property_sets[dataset.code] = load_canonical_smiles_set(csv_path)
    return property_sets


def build_sampled_grid_smiles(
    property_sets: dict[str, set[str]],
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
    random_seed: int = _DEFAULT_RANDOM_SEED,
) -> np.ndarray:
    union_smiles = sorted(set().union(*property_sets.values()))
    if sample_size > len(union_smiles):
        raise ValueError(
            f"Requested sample_size={sample_size} but only {len(union_smiles)} unique molecules exist in the union."
        )

    rng = np.random.default_rng(random_seed)
    sampled = rng.choice(np.array(union_smiles, dtype=object), size=sample_size, replace=False)
    rng.shuffle(sampled)
    return sampled


def sampled_smiles_to_grid(sampled_smiles: np.ndarray, grid_size: int = _DEFAULT_GRID_SIZE) -> np.ndarray:
    expected_size = grid_size * grid_size
    if len(sampled_smiles) != expected_size:
        raise ValueError(
            f"Expected exactly {expected_size} sampled molecules for a {grid_size}x{grid_size} grid; "
            f"received {len(sampled_smiles)}."
        )
    return sampled_smiles.reshape(grid_size, grid_size)


def build_membership_grid(sampled_grid: np.ndarray, property_smiles: set[str]) -> np.ndarray:
    return np.isin(sampled_grid, list(property_smiles))


def export_grid_mapping(sampled_grid: np.ndarray, output_csv: Path) -> None:
    records = []
    for row_idx in range(sampled_grid.shape[0]):
        for col_idx in range(sampled_grid.shape[1]):
            records.append(
                {
                    "row": row_idx,
                    "column": col_idx,
                    "canonical_smiles": sampled_grid[row_idx, col_idx],
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_csv, index=False)


def plot_membership_grid(
    membership_grid: np.ndarray,
    present_color: str,
    absent_color: str,
    save_path: Path,
    figsize: tuple[float, float] = (6.0, 6.0),
    dpi: int = 900,
) -> None:
    color_grid = membership_grid.astype(int)
    cmap = ListedColormap([absent_color, present_color])

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(color_grid, cmap=cmap, interpolation="nearest", aspect="equal", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, facecolor=_WHITE, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def generate_dataset_overlap_grids(
    data_dir: Path,
    output_dir: Path,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
    grid_size: int = _DEFAULT_GRID_SIZE,
    random_seed: int = _DEFAULT_RANDOM_SEED,
) -> dict[str, object]:
    property_sets = load_property_smiles_sets(data_dir)
    sampled_smiles = build_sampled_grid_smiles(
        property_sets=property_sets,
        sample_size=sample_size,
        random_seed=random_seed,
    )
    sampled_grid = sampled_smiles_to_grid(sampled_smiles, grid_size=grid_size)

    export_grid_mapping(sampled_grid, output_dir / "sampled_molecule_grid_map.csv")

    summary_rows = []
    for dataset in PROPERTY_DATASETS:
        membership_grid = build_membership_grid(sampled_grid, property_sets[dataset.code])
        present_count = int(membership_grid.sum())
        absent_count = int(membership_grid.size - present_count)

        red_white_path = output_dir / f"{dataset.code.lower()}_grid_red_white.png"
        red_blue_path = output_dir / f"{dataset.code.lower()}_grid_red_blue.png"

        plot_membership_grid(
            membership_grid=membership_grid,
            present_color=_PARITY_RED,
            absent_color=_WHITE,
            save_path=red_white_path,
        )
        plot_membership_grid(
            membership_grid=membership_grid,
            present_color=_PARITY_RED,
            absent_color=_PASTEL_BLUE,
            save_path=red_blue_path,
        )

        summary_rows.append(
            {
                "property_code": dataset.code,
                "property_label": dataset.label,
                "dataset_unique_molecules": len(property_sets[dataset.code]),
                "sampled_present_cells": present_count,
                "sampled_absent_cells": absent_count,
                "red_white_figure": str(red_white_path),
                "red_blue_figure": str(red_blue_path),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "dataset_overlap_grid_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "summary": summary_df,
        "summary_path": summary_path,
        "mapping_path": output_dir / "sampled_molecule_grid_map.csv",
        "sample_size": sample_size,
        "grid_size": grid_size,
        "random_seed": random_seed,
        "union_size": len(set().union(*property_sets.values())),
    }


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    repo_root = _default_repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Generate property-dataset overlap grids from the union of canonical SMILES in "
            "'All Datasets Accessible'."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "All Datasets Accessible",
        help="Directory containing the five property train CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "All Datasets Accessible" / "dataset_overlap_grids",
        help="Directory where the 10 grid figures and summary files will be written.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=_DEFAULT_SAMPLE_SIZE,
        help="Number of molecules to sample from the union of all canonicalized datasets.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=_DEFAULT_GRID_SIZE,
        help="Grid side length. sample-size must equal grid-size squared.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=_DEFAULT_RANDOM_SEED,
        help="Random seed used for reproducible union sampling and shuffling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_size != args.grid_size * args.grid_size:
        raise ValueError("sample-size must equal grid-size squared for a complete grid.")

    results = generate_dataset_overlap_grids(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        grid_size=args.grid_size,
        random_seed=args.random_seed,
    )

    print(f"Generated 10 figures in: {args.output_dir}")
    print(f"Union size: {results['union_size']}")
    print(f"Sample size: {results['sample_size']}")
    print(f"Grid size: {results['grid_size']} x {results['grid_size']}")
    print(f"Random seed: {results['random_seed']}")
    print(f"Grid mapping CSV: {results['mapping_path']}")
    print(f"Summary CSV: {results['summary_path']}")


if __name__ == "__main__":
    main()
