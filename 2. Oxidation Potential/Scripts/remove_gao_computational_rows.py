"""Remove provenance-confirmed Gao DFT rows from the current OP model inputs.

The redox values, transformed hydroquinone structures, and aligned descriptor
flags must all agree before any modeling file is changed. Re-running is safe.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Datasets"
RAW = DATA / "Raw Dataset"
CURATED = DATA / "Curated Dataset(s)"
TRAINABLE = DATA / "trainable_data"
TRAIN_FILE = TRAINABLE / "OP_chemprop_train_file.csv"
FEATURE_CSV = TRAINABLE / "OP_chemprop_feat_file.csv"
FEATURE_NPZ = TRAINABLE / "OP_chemprop_feat_file.npz"
ACCESSIBLE_COPY = ROOT.parent / "All Datasets Accessible" / TRAIN_FILE.name
REMOVAL_AUDIT = TRAINABLE / "Gao_computational_rows_removed.csv"
RAW_AUDIT = CURATED / "Gao_source_provenance_audit.csv"
DFT_DOIS = {"10.1039/c4sc03030c", "10.1039/d0se00687d"}


def canonical(smiles):
    if pd.isna(smiles):
        return None
    molecule = Chem.MolFromSmiles(str(smiles).strip())
    return Chem.MolToSmiles(molecule) if molecule is not None else None


def preserve_csv_rows(path, keep):
    lines = path.read_bytes().splitlines(keepends=True)
    if len(lines) != len(keep) + 1:
        raise ValueError(f"CSV physical row count mismatch: {path}")
    path.write_bytes(lines[0] + b"".join(line for line, retain in zip(lines[1:], keep) if retain))


def main():
    raw = pd.read_excel(RAW / "Gao_Data_Raw.xlsx")
    doi = raw["Ref DOI"].astype(str).str.strip().str.lower()
    raw["source_class"] = np.where(doi.isin(DFT_DOIS), "computational_DFT", "experimental")

    supplement = RAW / "Gao Experimental Data"
    exp_keys = set()
    exp_doi_values = set()
    for source in sorted(supplement.glob("*.csv")):
        table = pd.read_csv(source)
        key = "NeutralStateInchikey" if "NeutralStateInchikey" in table else "inchikey"
        exp_keys.update(table[key].dropna().astype(str).str.strip().str.upper())
        if {"DOI1", "RedoxPotential1(V)"}.issubset(table.columns):
            exp_doi_values.update(zip(
                table.DOI1.astype(str).str.strip().str.lower(),
                table["RedoxPotential1(V)"].round(4),
            ))
    raw["in_experimental_supplement_by_InChIKey"] = (
        raw.InChiKey.astype(str).str.strip().str.upper().isin(exp_keys)
    )
    raw["in_experimental_supplement_by_DOI_and_value"] = [
        (source_doi, round(float(value), 4)) in exp_doi_values
        for source_doi, value in zip(doi, raw["Redox Potential(V)"])
    ]
    raw.insert(0, "Gao_excel_row", np.arange(len(raw)) + 2)
    if raw.source_class.value_counts().to_dict() != {"computational_DFT": 586, "experimental": 291}:
        raise ValueError("Unexpected Gao raw source counts; review before filtering")

    hq = pd.read_csv(CURATED / "Gao_HQs_w_SMILES.csv")
    hq_doi = hq["Ref DOI"].astype(str).str.strip().str.lower()
    computed = hq[hq_doi.isin(DFT_DOIS)].copy()
    computed_pairs = set(zip(
        computed.canonical_SMILES.map(canonical),
        computed["E1/2 (SHE) (V)"].round(4),
    ))
    if not computed_pairs or None in {structure for structure, _ in computed_pairs}:
        raise ValueError("Cannot identify transformed Gao DFT structures")

    train = pd.read_csv(TRAIN_FILE)
    features = pd.read_csv(FEATURE_CSV)
    with np.load(FEATURE_NPZ) as archive:
        if archive.files != ["arr_0"]:
            raise ValueError("Unexpected descriptor NPZ format")
        descriptor_array = archive["arr_0"].copy()
    if len(train) != len(features) or len(train) != len(descriptor_array):
        raise ValueError("Training and descriptor row counts differ")
    if not np.array_equal(features.to_numpy(dtype=np.float32), descriptor_array):
        raise ValueError("Descriptor CSV and NPZ do not agree")
    if ACCESSIBLE_COPY.read_bytes() != TRAIN_FILE.read_bytes():
        raise ValueError("Accessible OP train copy differs from the modeling file")

    computed_match = np.array([
        (canonical(smiles), round(float(op), 4)) in computed_pairs
        for smiles, op in zip(train.SMILES, train.OP)
    ])
    descriptor_flag = (
        features.solvent_no_solvent.eq(1) & features.acidity_none.eq(1)
    ).to_numpy()
    if not np.array_equal(computed_match, descriptor_flag):
        raise ValueError("Gao DFT structure/value matches disagree with descriptor provenance")
    if computed_match.sum() not in (0, 330):
        raise ValueError("Unexpected number of Gao DFT training rows")

    removed = train.loc[computed_match].copy()
    removed.insert(0, "original_training_row_zero_based", np.flatnonzero(computed_match))
    removed["Gao_source_DOI"] = "10.1039/C4SC03030C"
    removed["classification_evidence"] = "transformed SMILES + redox value + no-solvent descriptors"
    if computed_match.any():
        keep = ~computed_match
        preserve_csv_rows(TRAIN_FILE, keep)
        preserve_csv_rows(FEATURE_CSV, keep)
        np.savez_compressed(FEATURE_NPZ, descriptor_array[keep])
        ACCESSIBLE_COPY.write_bytes(TRAIN_FILE.read_bytes())
        removed.to_csv(REMOVAL_AUDIT, index=False)
    raw.to_csv(RAW_AUDIT, index=False)

    print("Gao workbook: 291 experimental, 586 computational DFT")
    print("Removed OP rows by split:", removed.split.value_counts().to_dict())
    print("Retained OP rows by split:", train.loc[~computed_match, "split"].value_counts().to_dict())
    print("Wrote:", TRAIN_FILE, FEATURE_CSV, FEATURE_NPZ, ACCESSIBLE_COPY, RAW_AUDIT)


if __name__ == "__main__":
    main()
