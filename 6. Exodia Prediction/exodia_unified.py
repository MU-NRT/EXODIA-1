#!/usr/bin/env python3
"""
Unified Exodia dataset builder and predictor.

Builds a canonical SMILES union across the 5 property pipelines, preferring
experimental values when available and filling missing values with Chemprop
predictions. Vapor pressure is temperature-dependent and is resolved at runtime
for a user-provided temperature.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except Exception:
    Chem = None
    RDKIT_AVAILABLE = False


PROPERTY_ROOTS = {
    "VP": "1. Vapor Pressure",
    "OP": "2. Oxidation Potential",
    "PKA": "3. pKa",
    "MP": "4. Melting Point",
    "BP": "5. Boiling Point",
}

CURATED_FILES = {
    "VP": "Datasets/Curated Dataset(s)/SMILES_subset_c12_clean.csv",
    "OP": "Datasets/Curated Dataset(s)/Small_Organic_SMILES_to_OP_clean.csv",
    "PKA": "Datasets/Curated Dataset(s)/pka_data_clean.csv",
    "MP": "Datasets/Curated Dataset(s)/CleanedDataMP.csv",
    "BP": "Datasets/Curated Dataset(s)/Small_Organic_BP_clean.csv",
}

EXODIA_BEST_MODEL_FILENAMES = {
    "BP": "BP_best.pt",
    "MP": "MP_best.pt",
    "PKA": "pKa_best.pt",
    "OP": "OP_best.pt",
    "VP": "VP_best.pt",
}

OP_DESCRIPTOR_FEATURE_COLUMNS = [
    "solvent_aqueous",
    "solvent_non_aqueous",
    "solvent_mixed",
    "solvent_no_solvent",
    "acidity_acidic",
    "acidity_basic",
    "acidity_neutral",
    "acidity_none",
]

# Fallback descriptor row for OP when trainable feature files are unavailable.
# This corresponds to non-aqueous + neutral, the most common training context.
OP_DEFAULT_DESCRIPTOR_VECTOR = np.array(
    [0, 1, 0, 0, 0, 0, 1, 0],
    dtype=np.float32,
)

_OP_DESCRIPTOR_CACHE: dict[str, "OPDescriptorBundle"] = {}
KELVIN_OFFSET = 273.15


@dataclass
class ModelPaths:
    bp: Path
    mp: Path
    pka: Path
    op: Path
    vp: Path


@dataclass
class OPDescriptorBundle:
    feature_columns: list[str]
    by_canonical_smiles: dict[str, np.ndarray]
    default_vector: np.ndarray


def canonicalize_smiles(smiles: str) -> str | None:
    if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
        return None
    smi = str(smiles).strip()
    if not smi:
        return None
    if not RDKIT_AVAILABLE:
        # Fallback: keep the normalized raw SMILES string as-is.
        return smi
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _coalesce_smiles_column(df: pd.DataFrame) -> pd.Series:
    for col in ["SMILES", "smiles", "canonical_SMILES", "canonical_smiles"]:
        if col in df.columns:
            return df[col]
    raise ValueError("No SMILES-like column found.")


def _find_model_checkpoint(property_dir: Path) -> Path:
    best_weights = property_dir / "Best Model Weights"
    if best_weights.exists():
        pt_files = sorted(best_weights.glob("*.pt"))
        if pt_files:
            return pt_files[0]

    model_root = property_dir / "chemprop_model_output"
    candidates: list[tuple[int, Path]] = []
    if model_root.exists():
        for model_dir in model_root.iterdir():
            if not model_dir.is_dir() or not model_dir.name.startswith("model"):
                continue
            suffix = model_dir.name.replace("model", "")
            if not suffix.isdigit():
                continue
            best_pt = model_dir / "model_0" / "best.pt"
            if best_pt.exists():
                candidates.append((int(suffix), best_pt))

    if not candidates:
        raise FileNotFoundError(f"No model checkpoint found under: {property_dir}")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def resolve_model_paths(repo_root: Path) -> ModelPaths:
    return ModelPaths(
        bp=_find_model_checkpoint(repo_root / PROPERTY_ROOTS["BP"]),
        mp=_find_model_checkpoint(repo_root / PROPERTY_ROOTS["MP"]),
        pka=_find_model_checkpoint(repo_root / PROPERTY_ROOTS["PKA"]),
        op=_find_model_checkpoint(repo_root / PROPERTY_ROOTS["OP"]),
        vp=_find_model_checkpoint(repo_root / PROPERTY_ROOTS["VP"]),
    )


def resolve_exodia_best_model_paths(exodia_dir: Path) -> ModelPaths:
    best_dir = Path(exodia_dir) / "Best Models"
    if not best_dir.exists():
        raise FileNotFoundError(f"Best-model directory not found: {best_dir}")

    paths = {
        prop: best_dir / filename
        for prop, filename in EXODIA_BEST_MODEL_FILENAMES.items()
    }
    missing = [prop for prop, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing best-model checkpoint(s) for: {missing}. "
            f"Expected under: {best_dir}"
        )

    return ModelPaths(
        bp=paths["BP"],
        mp=paths["MP"],
        pka=paths["PKA"],
        op=paths["OP"],
        vp=paths["VP"],
    )


def resolve_default_model_paths(repo_root: Path, exodia_dir: Path | None = None) -> ModelPaths:
    resolved_exodia_dir = (
        Path(exodia_dir).resolve()
        if exodia_dir is not None
        else Path(__file__).resolve().parent
    )
    try:
        return resolve_exodia_best_model_paths(resolved_exodia_dir)
    except FileNotFoundError:
        return resolve_model_paths(repo_root)


def _read_curated(repo_root: Path, prop: str) -> pd.DataFrame:
    p = repo_root / PROPERTY_ROOTS[prop] / CURATED_FILES[prop]
    if not p.exists():
        raise FileNotFoundError(f"Missing curated file for {prop}: {p}")
    return pd.read_csv(p)


def _prepare_property_frames(repo_root: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}

    bp = _read_curated(repo_root, "BP").copy()
    bp["canonical_smiles"] = _coalesce_smiles_column(bp).map(canonicalize_smiles)
    bp = bp.dropna(subset=["canonical_smiles", "BP"])
    bp = bp.groupby("canonical_smiles", as_index=False)["BP"].median()
    frames["BP"] = bp

    mp = _read_curated(repo_root, "MP").copy()
    mp["canonical_smiles"] = _coalesce_smiles_column(mp).map(canonicalize_smiles)
    # Curated MP values are in Celsius; convert to Kelvin for consistency
    # with MP model predictions/trainable data.
    mp["MP"] = pd.to_numeric(mp["MP"], errors="coerce") + KELVIN_OFFSET
    mp = mp.dropna(subset=["canonical_smiles", "MP"])
    mp = mp.groupby("canonical_smiles", as_index=False)["MP"].median()
    frames["MP"] = mp

    pka = _read_curated(repo_root, "PKA").copy()
    pka["canonical_smiles"] = _coalesce_smiles_column(pka).map(canonicalize_smiles)
    pka = pka.dropna(subset=["canonical_smiles", "pka_value"])
    pka = pka.groupby("canonical_smiles", as_index=False)["pka_value"].median()
    pka = pka.rename(columns={"pka_value": "PKA"})
    frames["PKA"] = pka

    op = _read_curated(repo_root, "OP").copy()
    op["canonical_smiles"] = _coalesce_smiles_column(op).map(canonicalize_smiles)
    op = op.dropna(subset=["canonical_smiles", "OP"])
    op = op.groupby("canonical_smiles", as_index=False)["OP"].median()
    frames["OP"] = op

    vp = _read_curated(repo_root, "VP").copy()
    vp["canonical_smiles"] = _coalesce_smiles_column(vp).map(canonicalize_smiles)
    needed = ["canonical_smiles", "A", "B", "C", "TMIN", "TMAX"]
    for c in ["A", "B", "C", "TMIN", "TMAX"]:
        vp[c] = pd.to_numeric(vp[c], errors="coerce")
    vp = vp.dropna(subset=["canonical_smiles", "A", "B", "C"])
    vp = vp[needed].groupby("canonical_smiles", as_index=False).median()
    frames["VP"] = vp

    return frames


def _default_op_descriptor_bundle() -> OPDescriptorBundle:
    return OPDescriptorBundle(
        feature_columns=list(OP_DESCRIPTOR_FEATURE_COLUMNS),
        by_canonical_smiles={},
        default_vector=OP_DEFAULT_DESCRIPTOR_VECTOR.copy(),
    )


def _load_op_descriptor_bundle(repo_root: Path) -> OPDescriptorBundle:
    cache_key = str(Path(repo_root).resolve())
    if cache_key in _OP_DESCRIPTOR_CACHE:
        return _OP_DESCRIPTOR_CACHE[cache_key]

    op_root = repo_root / PROPERTY_ROOTS["OP"] / "Datasets" / "trainable_data"
    train_csv = op_root / "OP_chemprop_train_file.csv"
    feat_csv = op_root / "OP_chemprop_feat_file.csv"

    if not train_csv.exists() or not feat_csv.exists():
        bundle = _default_op_descriptor_bundle()
        _OP_DESCRIPTOR_CACHE[cache_key] = bundle
        return bundle

    train_df = pd.read_csv(train_csv)
    feat_df = pd.read_csv(feat_csv)

    n_rows = min(len(train_df), len(feat_df))
    if n_rows == 0:
        bundle = _default_op_descriptor_bundle()
        _OP_DESCRIPTOR_CACHE[cache_key] = bundle
        return bundle

    train_df = train_df.iloc[:n_rows].copy()
    feat_df = feat_df.iloc[:n_rows].copy()

    feature_columns = [c for c in OP_DESCRIPTOR_FEATURE_COLUMNS if c in feat_df.columns]
    if not feature_columns:
        feature_columns = feat_df.columns.tolist()

    if not feature_columns:
        bundle = _default_op_descriptor_bundle()
        _OP_DESCRIPTOR_CACHE[cache_key] = bundle
        return bundle

    for c in feature_columns:
        feat_df[c] = pd.to_numeric(feat_df[c], errors="coerce").fillna(0.0)

    feat_only = feat_df[feature_columns].copy()
    row_counts = feat_only.value_counts(dropna=False)
    if row_counts.empty:
        default_vector = OP_DEFAULT_DESCRIPTOR_VECTOR.copy()
        if len(default_vector) != len(feature_columns):
            default_vector = np.zeros(len(feature_columns), dtype=np.float32)
    else:
        mode_row = row_counts.index[0]
        if isinstance(mode_row, tuple):
            default_vector = np.asarray(mode_row, dtype=np.float32)
        else:
            default_vector = np.asarray([mode_row], dtype=np.float32)

    smiles_series = _coalesce_smiles_column(train_df)
    joined = feat_only.copy()
    joined["canonical_smiles"] = smiles_series.map(canonicalize_smiles)
    joined = joined.dropna(subset=["canonical_smiles"])

    by_smiles: dict[str, np.ndarray] = {}
    if not joined.empty:
        grouped = joined.groupby("canonical_smiles", as_index=False)[feature_columns].median()
        for _, row in grouped.iterrows():
            by_smiles[str(row["canonical_smiles"])] = row[feature_columns].to_numpy(dtype=np.float32)

    bundle = OPDescriptorBundle(
        feature_columns=feature_columns,
        by_canonical_smiles=by_smiles,
        default_vector=default_vector.astype(np.float32, copy=False),
    )
    _OP_DESCRIPTOR_CACHE[cache_key] = bundle
    return bundle


def _build_op_descriptor_matrix(smiles: list[str], repo_root: Path) -> np.ndarray:
    bundle = _load_op_descriptor_bundle(repo_root)
    if not smiles:
        return np.zeros((0, len(bundle.default_vector)), dtype=np.float32)

    vectors: list[np.ndarray] = []
    for s in smiles:
        cs = canonicalize_smiles(s)
        descriptor = bundle.by_canonical_smiles.get(cs) if cs else None
        if descriptor is None:
            descriptor = bundle.default_vector
        vectors.append(np.asarray(descriptor, dtype=np.float32))

    return np.vstack(vectors)


def _resolve_chemprop_command() -> str:
    cli = shutil.which("chemprop")
    if cli:
        return cli

    py = Path(sys.executable).resolve()
    candidates = [
        py.parent / "Scripts" / "chemprop.exe",  # Windows conda/venv layout
        py.parent / "chemprop",                  # POSIX venv layout
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    raise FileNotFoundError(
        "Could not find 'chemprop' executable. Install Chemprop in this environment "
        "or ensure it is available on PATH."
    )


def _predict_with_chemprop(
    smiles: list[str],
    model_path: Path,
    temperature_c: float | None = None,
    descriptors: np.ndarray | None = None,
) -> np.ndarray:
    if not smiles:
        return np.array([], dtype=float)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        inp = td_path / "input.csv"
        pred = td_path / "preds.csv"

        pd.DataFrame({"SMILES": smiles}).to_csv(inp, index=False)

        chemprop_cmd = _resolve_chemprop_command()
        cmd = [
            chemprop_cmd,
            "predict",
            "--test-path",
            str(inp),
            "--model-paths",
            str(model_path),
            "--preds-path",
            str(pred),
        ]

        descriptors_arr: np.ndarray | None = None
        if descriptors is not None:
            descriptors_arr = np.asarray(descriptors, dtype=np.float32)
            if descriptors_arr.ndim == 1:
                if len(smiles) != 1:
                    raise ValueError(
                        "Descriptor vector must be 2D when predicting multiple SMILES."
                    )
                descriptors_arr = descriptors_arr.reshape(1, -1)
            if descriptors_arr.shape[0] != len(smiles):
                raise ValueError(
                    "Descriptor rows must match number of SMILES "
                    f"({descriptors_arr.shape[0]} != {len(smiles)})."
                )

        if temperature_c is not None:
            temp_arr = np.full((len(smiles), 1), float(temperature_c), dtype=np.float32)
            descriptors_arr = temp_arr if descriptors_arr is None else np.hstack([descriptors_arr, temp_arr])

        if descriptors_arr is not None:
            npz_path = td_path / "descriptors.npz"
            np.savez(npz_path, descriptors_arr.astype(np.float32))
            cmd.extend(["--descriptors-path", str(npz_path)])

        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            stdout_tail = "\n".join(proc.stdout.strip().splitlines()[-20:])
            stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
            raise RuntimeError(
                "Chemprop prediction failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Exit code: {proc.returncode}\n"
                f"STDOUT (tail):\n{stdout_tail or '[empty]'}\n"
                f"STDERR (tail):\n{stderr_tail or '[empty]'}"
            )
        pred_df = pd.read_csv(pred)

        pred_cols = [
            c for c in pred_df.columns if c.lower() not in {"smiles", "smile"}
        ]
        if len(pred_cols) != 1:
            raise ValueError(f"Unexpected prediction columns from Chemprop: {pred_df.columns.tolist()}")
        return pred_df[pred_cols[0]].to_numpy(dtype=float)


def _antoine_log10_vp(a: float, b: float, c: float, t_c: float) -> float:
    return float(a) - float(b) / (float(t_c) + float(c))


def _build_union_smiles(frames: dict[str, pd.DataFrame], extra_smiles: Iterable[str] | None) -> pd.DataFrame:
    base = set()
    for f in frames.values():
        base.update(f["canonical_smiles"].dropna().astype(str).tolist())

    if extra_smiles is not None:
        for s in extra_smiles:
            cs = canonicalize_smiles(s)
            if cs:
                base.add(cs)

    return pd.DataFrame({"canonical_smiles": sorted(base)})


def build_unified_dataset(
    repo_root: Path,
    output_csv: Path,
    temperature_c: float,
    extra_smiles: Iterable[str] | None = None,
    model_paths: ModelPaths | None = None,
) -> pd.DataFrame:
    models = model_paths if model_paths is not None else resolve_default_model_paths(repo_root)
    frames = _prepare_property_frames(repo_root)

    union_df = _build_union_smiles(frames, extra_smiles)

    # Join experimental columns
    union_df = union_df.merge(frames["BP"].rename(columns={"BP": "BP_experimental"}), on="canonical_smiles", how="left")
    union_df = union_df.merge(frames["MP"].rename(columns={"MP": "MP_experimental"}), on="canonical_smiles", how="left")
    union_df = union_df.merge(frames["PKA"].rename(columns={"PKA": "PKA_experimental"}), on="canonical_smiles", how="left")
    union_df = union_df.merge(frames["OP"].rename(columns={"OP": "OP_experimental"}), on="canonical_smiles", how="left")
    union_df = union_df.merge(frames["VP"].add_prefix("VP_"), left_on="canonical_smiles", right_on="VP_canonical_smiles", how="left")

    # BP / MP / PKA / OP: experimental first, then prediction
    for prop, model_path in {
        "BP": models.bp,
        "MP": models.mp,
        "PKA": models.pka,
        "OP": models.op,
    }.items():
        exp_col = f"{prop}_experimental"
        pred_col = f"{prop}_predicted"
        val_col = f"{prop}_value"
        src_col = f"{prop}_source"

        union_df[pred_col] = np.nan
        missing_mask = union_df[exp_col].isna()
        missing_smiles = union_df.loc[missing_mask, "canonical_smiles"].tolist()
        op_descriptors = (
            _build_op_descriptor_matrix(missing_smiles, repo_root)
            if prop == "OP"
            else None
        )
        preds = _predict_with_chemprop(
            missing_smiles,
            model_path=model_path,
            descriptors=op_descriptors,
        )
        union_df.loc[missing_mask, pred_col] = preds

        union_df[val_col] = union_df[exp_col].combine_first(union_df[pred_col])
        union_df[src_col] = np.where(union_df[exp_col].notna(), "experimental", "predicted")

    # VP at requested temperature: experimental (Antoine) if coefficients apply, else prediction
    union_df["VP_experimental_log10"] = np.nan
    coeff_mask = (
        union_df["VP_A"].notna()
        & union_df["VP_B"].notna()
        & union_df["VP_C"].notna()
        & union_df["VP_TMIN"].notna()
        & union_df["VP_TMAX"].notna()
        & (float(temperature_c) >= union_df["VP_TMIN"])
        & (float(temperature_c) <= union_df["VP_TMAX"])
    )

    union_df.loc[coeff_mask, "VP_experimental_log10"] = union_df.loc[coeff_mask].apply(
        lambda r: _antoine_log10_vp(r["VP_A"], r["VP_B"], r["VP_C"], temperature_c),
        axis=1,
    )

    union_df["VP_predicted_log10"] = np.nan
    vp_missing = union_df["VP_experimental_log10"].isna()
    vp_missing_smiles = union_df.loc[vp_missing, "canonical_smiles"].tolist()
    vp_preds = _predict_with_chemprop(vp_missing_smiles, model_path=models.vp, temperature_c=temperature_c)
    union_df.loc[vp_missing, "VP_predicted_log10"] = vp_preds

    union_df["VP_value_log10"] = union_df["VP_experimental_log10"].combine_first(union_df["VP_predicted_log10"])
    union_df["VP_source"] = np.where(union_df["VP_experimental_log10"].notna(), "experimental", "predicted")
    union_df["VP_temperature_c"] = float(temperature_c)

    # Cleanup
    drop_cols = [c for c in ["VP_canonical_smiles"] if c in union_df.columns]
    if drop_cols:
        union_df = union_df.drop(columns=drop_cols)

    union_df = union_df.sort_values("canonical_smiles").reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    union_df.to_csv(output_csv, index=False)
    return union_df


def parse_args() -> argparse.Namespace:
    default_output = Path("6. Exodia Prediction") / "Datasets" / "unified_dataset_with_experimental_and_predictions.csv"

    p = argparse.ArgumentParser(description="Build unified Exodia dataset with experimental-first values and model backfill.")
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1], help="Repository root path.")
    p.add_argument("--temperature-c", type=float, required=True, help="Temperature in Celsius for VP resolution.")
    p.add_argument("--output", type=Path, default=default_output, help="Output CSV path.")
    p.add_argument("--extra-smiles-csv", type=Path, default=None, help="Optional CSV with additional SMILES to include.")
    p.add_argument("--extra-smiles-col", type=str, default="SMILES", help="SMILES column name in --extra-smiles-csv.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    extra_smiles = None
    if args.extra_smiles_csv is not None:
        xdf = pd.read_csv(args.extra_smiles_csv)
        if args.extra_smiles_col not in xdf.columns:
            raise ValueError(f"Column '{args.extra_smiles_col}' not found in {args.extra_smiles_csv}")
        extra_smiles = xdf[args.extra_smiles_col].dropna().astype(str).tolist()

    df = build_unified_dataset(
        repo_root=repo_root,
        output_csv=args.output,
        temperature_c=args.temperature_c,
        extra_smiles=extra_smiles,
    )

    if not RDKIT_AVAILABLE:
        print("Warning: RDKit not available; SMILES canonicalization fallback is raw-string based.")

    print(f"Wrote unified dataset: {args.output}")
    print(f"Rows: {len(df)}")
    print("Model checkpoints used:")
    paths = resolve_default_model_paths(repo_root)
    print(f"  BP:  {paths.bp}")
    print(f"  MP:  {paths.mp}")
    print(f"  PKA: {paths.pka}")
    print(f"  OP:  {paths.op}")
    print(f"  VP:  {paths.vp}")


if __name__ == "__main__":
    main()
