"""Read-only provenance checks for the retained OP rows.

Uses the original quinone-to-hydroquinone conversion in Data_Carpentry_OP.ipynb
to compare Gao's experimental supplemental tables with the current OP splits.
"""

import contextlib
import io
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Datasets"
CURATED = DATA / "Curated Dataset(s)"
SUPPLEMENT = DATA / "Raw Dataset" / "Gao Experimental Data"


def canonical(smiles):
    if pd.isna(smiles):
        return None
    molecule = Chem.MolFromSmiles(str(smiles).replace("\xa0", "").strip())
    return Chem.MolToSmiles(molecule) if molecule is not None else None


def original_reduction_function():
    notebook = json.loads((ROOT / "Exploratory Work" / "Data_Carpentry_OP.ipynb").read_text(encoding="utf-8"))
    source = next(
        ("".join(cell.get("source", [])) for cell in notebook["cells"]
         if "def reduce_quinone_stepwise_with_radicals" in "".join(cell.get("source", []))),
        None,
    )
    if source is None:
        raise ValueError("The original Gao quinone conversion was not found in Data_Carpentry_OP.ipynb")
    # Omit the notebook's molecule image creation; retain its chemistry steps.
    source = source[: source.index("    # Draw before/after side by side")] + "    return newmol\n"
    namespace = {}
    exec(source, namespace)
    return namespace["reduce_quinone_stepwise_with_radicals"]


def transformed(smiles, reduction):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            reduced = reduction(str(smiles).replace("\xa0", "").strip())
        return canonical(Chem.MolToSmiles(reduced)) if reduced is not None else None
    except Exception:
        return None


def source_pairs(frame, smiles_column, value_column, reduction=None):
    original = frame[smiles_column].map(canonical)
    value = frame[value_column].round(4)
    exact = {(smiles, potential) for smiles, potential in zip(original, value) if smiles is not None}
    if reduction is None:
        return exact, set()
    transformed_smiles = frame[smiles_column].map(lambda smiles: transformed(smiles, reduction))
    reduced_pairs = {(smiles, potential) for smiles, potential in zip(transformed_smiles, value) if smiles is not None}
    return exact, reduced_pairs


def main():
    train = pd.read_csv(DATA / "trainable_data" / "OP_chemprop_train_file.csv")
    train["canonical"] = train.SMILES.map(canonical)
    train["value"] = train.OP.round(4)
    observed = list(zip(train.canonical, train.value))

    reduction = original_reduction_function()
    sources = {}
    for number in (1, 2, 3):
        name = f"dataset3-{number}.csv"
        table = pd.read_csv(SUPPLEMENT / name)
        smiles = "smiles" if number == 3 else "NeutralSMILES"
        value = "Redoxpotential1e(V)" if number == 3 else "RedoxPotential1(V)"
        sources[name] = source_pairs(table, smiles, value, reduction)

    gao_raw = pd.read_excel(DATA / "Raw Dataset" / "Gao_Data_Raw.xlsx")
    computational_dois = {"10.1039/c4sc03030c", "10.1039/d0se00687d"}
    gao_experimental = gao_raw.loc[
        ~gao_raw["Ref DOI"].astype(str).str.strip().str.lower().isin(computational_dois)
    ]
    sources["Gao_Data_Raw.xlsx experimental rows"] = source_pairs(
        gao_experimental, "SMILES", "Redox Potential(V)", reduction
    )

    weinberg = pd.read_csv(CURATED / "Weinberg_w_SMILES.csv")
    # The active model uses these offsets for z/Z and Y, although the older
    # Weinberg curated export uses +0.800 V for both reference codes.
    weinberg["model_potential"] = weinberg["E1/2 (SHE) (V)"].copy()
    for code, offset in {"z": 0.099, "Z": 0.099, "Y": 0.158}.items():
        mask = weinberg["Ref Electrode"].eq(code)
        weinberg.loc[mask, "model_potential"] = weinberg.loc[mask, "E1/2 (V)"] + offset
    sources["Weinberg_w_SMILES.csv (model reference offsets)"] = source_pairs(
        weinberg, "canonical_SMILES", "model_potential"
    )
    roth = pd.read_csv(CURATED / "Roth_positive_pubchem_SMILES_to_OP_no_splits_candidate.csv")
    roth["OP_vs_SHE"] = roth.OP + 0.241
    sources["Roth_positive_pubchem_SMILES_to_OP_no_splits_candidate.csv (+0.241 V SCE-to-SHE)"] = source_pairs(
        roth, "SMILES", "OP_vs_SHE"
    )

    results = []
    matches_by_source = {}
    for name, (original, reduced) in sources.items():
        match_original = [pair in original for pair in observed]
        match_reduced = [pair in reduced for pair in observed]
        matches = [a or b for a, b in zip(match_original, match_reduced)]
        matches_by_source[name] = matches
        subset = train.loc[matches]
        results.append((name, len(original), len(reduced), len(subset), subset.canonical.nunique(),
                        subset.split.value_counts().to_dict(), sum(match_original), sum(match_reduced)))
    for result in results:
        print(result)

    groups = {
        "Gao experimental": [name for name in sources if name.startswith("dataset3-") or name.startswith("Gao_Data_Raw")],
        "Weinberg": [name for name in sources if name.startswith("Weinberg")],
        "Roth": [name for name in sources if name.startswith("Roth")],
    }
    group_matches = {}
    for group, names in groups.items():
        mask = [any(matches_by_source[name][i] for name in names) for i in range(len(train))]
        group_matches[group] = mask
        subset = train.loc[mask]
        print("SOURCE TOTAL", group, len(subset), "unique molecules", subset.canonical.nunique(),
              "by split", subset.split.value_counts().to_dict())
    print("Rows assigned to multiple groups", sum(
        sum(mask[i] for mask in group_matches.values()) > 1 for i in range(len(train))
    ))

    union = set().union(*(original | reduced for original, reduced in sources.values()))
    unmatched = train.loc[[pair not in union for pair in observed]]
    print("total", len(train), "matched any exact source pair", len(train) - len(unmatched),
          "unmatched", len(unmatched))
    print("unmatched by split", unmatched.split.value_counts().to_dict())
    weinberg_structures = {structure for structure, _ in sources[
        "Weinberg_w_SMILES.csv (model reference offsets)"
    ][0]}
    roth_structures = {structure for structure, _ in sources[
        "Roth_positive_pubchem_SMILES_to_OP_no_splits_candidate.csv (+0.241 V SCE-to-SHE)"
    ][0]}
    gao_structures = {structure for name, (original, reduced) in sources.items()
                      if name.startswith("dataset3-") for structure, _ in original | reduced}
    print("unmatched structure overlaps:", {
        "Weinberg": unmatched.canonical.isin(weinberg_structures).sum(),
        "Roth": unmatched.canonical.isin(roth_structures).sum(),
        "Gao": unmatched.canonical.isin(gao_structures).sum(),
    })
    external = weinberg_structures | roth_structures | gao_structures
    print("unmatched structures absent from all source tables:")
    print(unmatched.loc[~unmatched.canonical.isin(external), ["SMILES", "OP", "split"]].to_string(index=True))
    print("unmatched Gao structure:")
    print(unmatched.loc[unmatched.canonical.isin(gao_structures), ["SMILES", "OP", "split"]].to_string(index=True))


if __name__ == "__main__":
    main()
