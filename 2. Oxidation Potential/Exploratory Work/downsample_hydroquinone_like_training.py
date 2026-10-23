from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR / "Datasets" / "trainable_data" / "OP_chemprop_train_file.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "Datasets" / "trainable_data" / "OP_chemprop_train_file_hq_scaffold_downsampled_cap60.csv"
DEFAULT_AUDIT = PROJECT_DIR / "Datasets" / "trainable_data" / "OP_chemprop_train_file_hq_scaffold_downsampled_cap60_audit.csv"

SMILES_COL = "SMILES"
TARGET_COL = "OP"
SPLIT_COL = "split"
TRAIN_SPLIT = "train"


def count_phenolic_oh(mol: Chem.Mol) -> int:
    phenol = Chem.MolFromSmarts("[c][O;H1]")
    return len({match[1] for match in mol.GetSubstructMatches(phenol)})


def evenly_spaced_indices(n_rows: int, keep_count: int) -> list[int]:
    if keep_count >= n_rows:
        return list(range(n_rows))
    if keep_count <= 1:
        return [0]

    raw = [round(i * (n_rows - 1) / (keep_count - 1)) for i in range(keep_count)]
    deduped = sorted(set(raw))

    if len(deduped) == keep_count:
        return deduped

    remaining = [idx for idx in range(n_rows) if idx not in deduped]
    while len(deduped) < keep_count and remaining:
        deduped.append(remaining.pop(len(remaining) // 2))
        deduped = sorted(set(deduped))
    return deduped[:keep_count]


def annotate_dataset(df: pd.DataFrame, phenol_threshold: int) -> pd.DataFrame:
    records: list[dict] = []
    for row in df.itertuples(index=False):
        smiles = getattr(row, SMILES_COL)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            records.append(
                {
                    **row._asdict(),
                    "canonical_smiles": None,
                    "murcko_scaffold": None,
                    "phenolic_oh_count": 0,
                    "hydroquinone_like": False,
                    "rdkit_valid": False,
                }
            )
            continue

        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        phenolic_oh_count = count_phenolic_oh(mol)
        records.append(
            {
                **row._asdict(),
                "canonical_smiles": canonical_smiles,
                "murcko_scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=mol),
                "phenolic_oh_count": phenolic_oh_count,
                "hydroquinone_like": phenolic_oh_count >= phenol_threshold,
                "rdkit_valid": True,
            }
        )
    return pd.DataFrame(records)


def downsample_training_df(
    df: pd.DataFrame,
    max_per_scaffold: int,
    phenol_threshold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    annotated = annotate_dataset(df, phenol_threshold=phenol_threshold)
    annotated["kept_in_downsampled"] = True
    annotated["keep_reason"] = "unchanged"

    train_mask = annotated[SPLIT_COL] == TRAIN_SPLIT
    hq_train_mask = train_mask & annotated["hydroquinone_like"]

    hq_train = annotated.loc[hq_train_mask].copy()
    non_hq_or_holdout = annotated.loc[~hq_train_mask].copy()

    kept_groups: list[pd.DataFrame] = []
    dropped_groups: list[pd.DataFrame] = []

    for scaffold, group in hq_train.groupby("murcko_scaffold", dropna=False, sort=False):
        ordered = group.sort_values(TARGET_COL).reset_index()
        keep_n = min(len(ordered), max_per_scaffold)
        keep_idx = set(evenly_spaced_indices(len(ordered), keep_n))

        kept = ordered.loc[ordered.index.isin(keep_idx)].copy()
        dropped = ordered.loc[~ordered.index.isin(keep_idx)].copy()

        kept["keep_reason"] = f"hydroquinone_like_kept_scaffold_cap_{max_per_scaffold}"
        dropped["keep_reason"] = f"hydroquinone_like_dropped_scaffold_cap_{max_per_scaffold}"
        dropped["kept_in_downsampled"] = False

        kept_groups.append(kept.drop(columns="index"))
        dropped_groups.append(dropped.drop(columns="index"))

    kept_hq = pd.concat(kept_groups, ignore_index=True) if kept_groups else annotated.iloc[0:0].copy()
    dropped_hq = pd.concat(dropped_groups, ignore_index=True) if dropped_groups else annotated.iloc[0:0].copy()

    final_annotated = pd.concat([non_hq_or_holdout, kept_hq, dropped_hq], ignore_index=True)
    final_annotated = final_annotated.sort_values(
        [SPLIT_COL, "kept_in_downsampled", "hydroquinone_like", TARGET_COL, SMILES_COL],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)

    output_df = (
        final_annotated.loc[final_annotated["kept_in_downsampled"], [SMILES_COL, TARGET_COL, SPLIT_COL]]
        .reset_index(drop=True)
    )

    summary_rows = []
    train_original = annotated.loc[train_mask]
    train_final = final_annotated.loc[(final_annotated["kept_in_downsampled"]) & (final_annotated[SPLIT_COL] == TRAIN_SPLIT)]
    for label, frame in [("original_train", train_original), ("downsampled_train", train_final)]:
        summary_rows.append(
            {
                "dataset": label,
                "rows": len(frame),
                "hydroquinone_like_rows": int(frame["hydroquinone_like"].sum()),
                "non_hydroquinone_like_rows": int((~frame["hydroquinone_like"]).sum()),
                "hydroquinone_like_fraction": float(frame["hydroquinone_like"].mean()) if len(frame) else 0.0,
                "unique_hq_scaffolds": int(frame.loc[frame["hydroquinone_like"], "murcko_scaffold"].nunique(dropna=False)),
                "target_min": float(frame[TARGET_COL].min()) if len(frame) else float("nan"),
                "target_max": float(frame[TARGET_COL].max()) if len(frame) else float("nan"),
                "target_mean": float(frame[TARGET_COL].mean()) if len(frame) else float("nan"),
                "target_median": float(frame[TARGET_COL].median()) if len(frame) else float("nan"),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return output_df, final_annotated, summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downsample hydroquinone-like molecules in the training split using a scaffold cap."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--max-per-scaffold", type=int, default=60)
    parser.add_argument(
        "--phenol-threshold",
        type=int,
        default=2,
        help="Mark molecules as hydroquinone-like when they contain at least this many phenolic OH groups.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    audit_path = Path(args.audit_output)

    source_df = pd.read_csv(input_path)
    output_df, audit_df, summary_df = downsample_training_df(
        source_df,
        max_per_scaffold=args.max_per_scaffold,
        phenol_threshold=args.phenol_threshold,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    audit_df.to_csv(audit_path, index=False)

    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Audit file: {audit_path}")
    print(summary_df.to_string(index=False))

    dropped_hq = audit_df[
        (audit_df[SPLIT_COL] == TRAIN_SPLIT)
        & (audit_df["hydroquinone_like"])
        & (~audit_df["kept_in_downsampled"])
    ]
    if not dropped_hq.empty:
        scaffold_counts = (
            dropped_hq["murcko_scaffold"]
            .fillna("<no scaffold>")
            .value_counts()
            .rename_axis("murcko_scaffold")
            .reset_index(name="dropped_rows")
        )
        print("\nDropped hydroquinone-like rows by scaffold:")
        print(scaffold_counts.to_string(index=False))


if __name__ == "__main__":
    main()
