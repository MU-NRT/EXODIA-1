from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
import pubchempy as pcp
from rdkit import Chem


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_INPUT = PROJECT_DIR / "Datasets" / "Raw Dataset" / "Roth_Data_Raw.csv"
RAW_OUTPUT = PROJECT_DIR / "Datasets" / "Raw Dataset" / "Roth_positive_pubchem_lookup.csv"
CURATED_OUTPUT = (
    PROJECT_DIR
    / "Datasets"
    / "Curated Dataset(s)"
    / "Roth_positive_pubchem_SMILES_to_OP_no_splits_candidate.csv"
)

def canonicalize_smiles(smiles: Optional[str]) -> Optional[str]:
    if smiles is None or pd.isna(smiles):
        return None
    smiles = str(smiles).strip()
    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


@lru_cache(maxsize=None)
def query_pubchem(identifier: str) -> Optional[dict]:
    identifier = str(identifier).strip()
    if not identifier or identifier.lower() in {"nan", "none"}:
        return None

    compounds = pcp.get_compounds(identifier, "name")
    if not compounds:
        return None

    compound = compounds[0]
    raw_smiles = (
        getattr(compound, "smiles", None)
        or getattr(compound, "connectivity_smiles", None)
        or getattr(compound, "canonical_smiles", None)
        or getattr(compound, "isomeric_smiles", None)
    )

    return {
        "CID": compound.cid,
        "raw_smiles": raw_smiles,
    }


def lookup_smiles(cas_number: Optional[str], molecule_name: Optional[str]) -> dict:
    for source, identifier in (("cas_number", cas_number), ("name", molecule_name)):
        if identifier is None or pd.isna(identifier):
            continue

        identifier = str(identifier).strip()
        if not identifier:
            continue

        try:
            result = query_pubchem(identifier)
        except Exception:
            result = None

        if not result:
            continue

        raw_smiles = result.get("raw_smiles")
        canonical_smiles = canonicalize_smiles(raw_smiles)
        if canonical_smiles is None:
            continue

        return {
            "SMILES": canonical_smiles,
            "pubchem_cid": result.get("CID"),
            "lookup_source": source,
        }

    return {
        "SMILES": None,
        "pubchem_cid": None,
        "lookup_source": None,
    }


def build_exports() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = pd.read_csv(RAW_INPUT)
    raw_df["E_exp_V_vs_SCE"] = pd.to_numeric(raw_df["E_exp_V_vs_SCE"], errors="coerce")

    positive_df = raw_df.loc[raw_df["E_exp_V_vs_SCE"] > 0, ["name", "cas_number", "E_exp_V_vs_SCE"]].copy()
    positive_df = positive_df.reset_index(drop=True)

    lookup_records = [
        lookup_smiles(cas_number=row.cas_number, molecule_name=row.name)
        for row in positive_df.itertuples(index=False)
    ]
    lookup_df = pd.DataFrame(lookup_records)

    review_df = pd.concat([positive_df, lookup_df], axis=1)

    curated_df = (
        review_df.loc[review_df["SMILES"].notna(), ["SMILES", "E_exp_V_vs_SCE"]]
        .rename(columns={"E_exp_V_vs_SCE": "OP"})
        .reset_index(drop=True)
    )

    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    CURATED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(RAW_OUTPUT, index=False)
    curated_df.to_csv(CURATED_OUTPUT, index=False)

    return review_df, curated_df


def main() -> None:
    review_df, curated_df = build_exports()

    positive_count = len(review_df)
    resolved_count = int(review_df["SMILES"].notna().sum())
    unresolved_count = positive_count - resolved_count

    print(f"Positive-voltage Roth rows: {positive_count}")
    print(f"Resolved PubChem SMILES: {resolved_count}")
    print(f"Unresolved rows: {unresolved_count}")
    print(f"Review export: {RAW_OUTPUT}")
    print(f"Curated-shape export: {CURATED_OUTPUT}")

    if unresolved_count:
        unresolved_preview = review_df.loc[
            review_df["SMILES"].isna(), ["name", "cas_number", "E_exp_V_vs_SCE"]
        ].head(10)
        print("\nFirst unresolved rows:")
        preview_text = unresolved_preview.to_string(index=False)
        print(preview_text.encode("ascii", errors="backslashreplace").decode("ascii"))


if __name__ == "__main__":
    main()
