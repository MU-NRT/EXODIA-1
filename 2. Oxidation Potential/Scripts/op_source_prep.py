"""Load the three OP source datasets with row-level preparation provenance.

The cached Weinberg and Roth SMILES lookups make notebook runs reproducible
without querying PubChem. Gao quinones use the transformation already stored
in the exploratory notebook (via ``trace_op_sources``).
"""

from pathlib import Path

import pandas as pd

from trace_op_sources import canonical, original_reduction_function, transformed


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Datasets" / "Raw Dataset"
CURATED = ROOT / "Datasets" / "Curated Dataset(s)"
GAO_DFT_DOIS = {"10.1039/c4sc03030c", "10.1039/d0se00687d"}
WEINBERG_ALLOWED_REFS = {"x", "y", "z", "n", "s", "f"}


def _first_potential(value):
    """Reproduce the Weinberg notebook's first-wave parsing."""
    cleaned = pd.Series([value]).astype(str).str.replace(
        r"(?i)near", "", regex=True
    ).str.replace(r"[^0-9\.\,\-\+]", "", regex=True).iloc[0].strip()
    try:
        return float(cleaned.split(",")[0])
    except (ValueError, IndexError):
        return float("nan")


def _record(dataset, source_file, source_row, raw_smiles, raw_potential,
            smiles, potential, status, detail="", reported_condition=None):
    return {
        "source_dataset": dataset,
        "source_file": source_file,
        "source_row": source_row,
        "raw_SMILES": raw_smiles,
        "raw_potential_V": raw_potential,
        "SMILES": smiles,
        "OP": potential,
        "preparation_status": status,
        "detail": detail,
        "reported_condition": reported_condition,
    }


def _gao_condition(doi, ph):
    """Identify repeat rows from one Gao reference under its reported pH.

    Gao gives pH or an acid note, rather than a complete solvent recipe. The
    reference is included so equal pH values from different papers are not
    assumed to describe the same solvent system.
    """
    if pd.isna(ph):
        return None
    value = str(ph).strip().lower()
    if not value or value == "nan":
        return None
    if value.endswith(".0") and value[:-2].isdigit():
        value = value[:-2]
    if value.startswith("acid"):
        value = "acid (unspecified)"
    value = value.replace("surfuric", "sulfuric").replace("sulfuric acid", "h2so4")
    value = value.replace("in ", "").replace(" ", "")
    ref = str(doi).strip().lower().removeprefix("https://doi.org/")
    if not ref or ref == "nan":
        return None
    return f"Gao|doi={ref}|ph={value}"


def load_weinberg():
    raw = pd.read_excel(RAW / "Weinberg_Data_Raw.xlsx", sheet_name="Sheet1")
    cached = pd.read_csv(CURATED / "Weinberg_w_SMILES.csv").set_index("Unnamed: 0")
    records = []
    for index, row in raw.iterrows():
        text = str(row["E1/2 (V)"])
        ref = str(row["Ref Electrode"]).strip()
        detail = f"{row['Compound']} | electrode={ref}"
        condition = f"Weinberg|solvent_code={str(row['Solvent System']).strip()}"
        if any(marker in text for marker in ("Ep/2", "Ep", "Ed")):
            records.append(_record("Weinberg", "Weinberg_Data_Raw.xlsx", index + 2,
                                   None, text, None, None, "non_half_wave_potential", detail))
            continue
        measured = _first_potential(row["E1/2 (V)"])
        if pd.isna(measured):
            records.append(_record("Weinberg", "Weinberg_Data_Raw.xlsx", index + 2,
                                   None, text, None, None, "invalid_potential", detail))
            continue
        if ref.lower() not in WEINBERG_ALLOWED_REFS:
            records.append(_record("Weinberg", "Weinberg_Data_Raw.xlsx", index + 2,
                                   None, measured, None, None, "unsupported_reference", detail))
            continue
        if index not in cached.index:
            records.append(_record("Weinberg", "Weinberg_Data_Raw.xlsx", index + 2,
                                   None, measured, None, None, "unresolved_smiles", detail))
            continue
        saved = cached.loc[index]
        smiles = canonical(saved["canonical_SMILES"])
        if smiles is None:
            status, potential = "invalid_smiles", None
        else:
            # These z/Z and Y offsets reproduce the existing model53 targets.
            # The older Weinberg curated export instead used +0.800 V.
            if ref in {"z", "Z"}:
                potential = measured + 0.099
            elif ref == "Y":
                potential = measured + 0.158
            else:
                potential = float(saved["E1/2 (SHE) (V)"])
            status = "ready"
        records.append(_record("Weinberg", "Weinberg_Data_Raw.xlsx", index + 2,
                                   saved["SMILES"], measured, smiles, potential, status,
                                   detail, condition))
    return pd.DataFrame(records)


def load_roth():
    raw = pd.read_csv(RAW / "Roth_Data_Raw.csv")
    lookup = pd.read_csv(RAW / "Roth_positive_pubchem_lookup.csv")
    positive = raw["E_exp_V_vs_SCE"].gt(0)
    if int(positive.sum()) != len(lookup):
        raise ValueError("Roth cached SMILES lookup does not align with positive source rows")
    records = []
    lookup_index = 0
    for index, row in raw.iterrows():
        measured = float(row["E_exp_V_vs_SCE"])
        detail = f"{row['name']} | CAS={row['cas_number']}"
        if measured <= 0:
            records.append(_record("Roth", "Roth_Data_Raw.csv", index + 2,
                                   None, measured, None, None, "nonpositive_potential", detail))
            continue
        saved = lookup.iloc[lookup_index]
        lookup_index += 1
        if abs(float(saved["E_exp_V_vs_SCE"]) - measured) > 1e-6 or saved["name"] != row["name"]:
            raise ValueError(f"Roth cached lookup mismatch at raw row {index + 2}")
        smiles = canonical(saved["SMILES"])
        status = "ready" if smiles is not None else "unresolved_smiles"
        records.append(_record("Roth", "Roth_Data_Raw.csv", index + 2,
                               saved["SMILES"], measured, smiles,
                               measured + 0.241 if smiles is not None else None,
                               status, detail))
    return pd.DataFrame(records)


def load_gao():
    reduction = original_reduction_function()
    records = []
    workbook = pd.read_excel(RAW / "Gao_Data_Raw.xlsx", sheet_name="Gao")
    for index, row in workbook.iterrows():
        doi = str(row["Ref DOI"]).strip()
        measured = float(row["Redox Potential(V)"])
        detail = f"DOI={doi} | pH={row['pH']}"
        condition = _gao_condition(doi, row["pH"])
        if doi.lower() in GAO_DFT_DOIS:
            records.append(_record("Gao", "Gao_Data_Raw.xlsx", index + 2,
                                   row["SMILES"], measured, None, None,
                                   "computational_dft", detail))
            continue
        smiles = transformed(row["SMILES"], reduction)
        status = "ready" if smiles is not None else "quinone_conversion_failed"
        records.append(_record("Gao", "Gao_Data_Raw.xlsx", index + 2,
                               row["SMILES"], measured, smiles,
                               measured if smiles is not None else None, status,
                               detail, condition))

    supplement = RAW / "Gao Experimental Data"
    for number in (1, 2, 3):
        filename = f"dataset3-{number}.csv"
        table = pd.read_csv(supplement / filename)
        smiles_column = "smiles" if number == 3 else "NeutralSMILES"
        value_column = "Redoxpotential1e(V)" if number == 3 else "RedoxPotential1(V)"
        for index, row in table.iterrows():
            measured = float(row[value_column])
            smiles = transformed(row[smiles_column], reduction)
            status = "ready" if smiles is not None else "quinone_conversion_failed"
            detail = str(row.get("DOI1", row.get("name", "")))
            if number == 1:
                condition = _gao_condition(row.get("DOI1"), row.get("pH_1"))
            elif number == 2:
                ph = row.get("pH1")
                condition = _gao_condition(row.get("DOI1"),
                                           row.get("Note1") if pd.isna(ph) else ph)
            else:
                # Gao Experimental Data/manuscript_clean.pdf, section 2.2.3,
                # identifies all 336 Q/Q.- one-electron measurements as DMF.
                condition = "Gao|solvent=DMF|couple=Q/Qradical"
            records.append(_record("Gao", filename, index + 2,
                                   row[smiles_column], measured, smiles,
                                   measured if smiles is not None else None, status,
                                   detail, condition))
    return pd.DataFrame(records)


def build_op_source_audit():
    """Return every source row plus the rows eligible for chemistry filtering."""
    audit = pd.concat([load_weinberg(), load_roth(), load_gao()], ignore_index=True)
    ready = audit.loc[audit.preparation_status.eq("ready")].copy()
    ready["OP"] = ready.OP.round(4)
    if ready[["SMILES", "OP"]].isna().any().any():
        raise ValueError("Ready source records contain missing structures or targets")
    return audit, ready
