from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem

from plot_top_functional_groups_in_grid_file import plot_top_functional_groups_in_grid


SMARTS_GROUPS = {
    "Vinylic Carbon": "[C!R]=[C!R]",
    "Alkyne Carbon": "[C!R]#[C!R]",
    "Carbonyl": "[CX3,#14]=[OX1]",
    "Ether": "[OD2]([#6!R,#14])[#6!R,#14]",
    "Primary Amine": "[NX3H2][C,#14]",
    "Secondary Amine": "[NX3H1]([C,#14])([C,#14])",
    "Tertiary Amine": "[NX3]([C!R,#14])([C!R,#14])([C!R,#14])",
    "Azo Nitrogen": "[NX2]=N",
    "Imine": "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]",
    "Nitrile": "[NX1]#[CX2]",
    "Nitroso": "[NX2]=[OX1]",
    "Hydroxyl": "[OX2H]-[#6,#14]",
    "Thiol/Sulfide/Disulfide": "[SX2]",
    "Sulfoxide": "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]",
    "Sulfate": "[$([#16X4](=[OX1])(=[OX1])([OX2H,OX1H0-])[OX2][#6])]",
    "Halogen": "[F,Cl,Br,I]-[#6,#14]",
    "Non-alkyl Ring": "[R;!$([#6X4,!#6])]",
    "Lone pair Atoms": "[O,N;!+0,F,Cl,Br,I,S]",
}
SMARTS_PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in SMARTS_GROUPS.items()}


def _count_total_functional_groups(smiles: str, patterns: dict) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0

    claimed_atoms = set()
    ring_atoms = set()
    total = 0

    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if any(
            mol.GetAtomWithIdx(i).GetAtomicNum() != 6
            or mol.GetAtomWithIdx(i).GetHybridization().name != "SP3"
            for i in ring
        ):
            total += 1
            ring_atoms.update(ring)

    for name, patt in patterns.items():
        if name == "Non-alkyl Ring":
            continue
        matches = mol.GetSubstructMatches(patt)
        for match in matches:
            if any(atom in claimed_atoms for atom in match) or all(atom in ring_atoms for atom in match):
                continue
            total += 1
            claimed_atoms.update(match)

    return total


def run_dataprep_plots(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    property_col: str = "BP",
    non_h_atom_col: str = "non_H_atom_count",
    property_label: str = "Property",
    property_units: str = "",
    property_hist_bins: int = 50,
    top_n: int = 30,
    max_per_row: int = 6,
    include_reactive_group_distribution: bool = True,
    bp_col: str | None = None,
) -> None:
    # Backward-compatible support for older notebook calls.
    if bp_col is not None:
        property_col = bp_col

    units_suffix = f" [{property_units}]" if property_units else ""

    freq = df[non_h_atom_col].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    plt.bar(freq.index, freq.values, color="skyblue", edgecolor="black")
    plt.xlabel("Number of Non-Hydrogen Atoms", fontsize=12)
    plt.ylabel("Number of Compounds", fontsize=12)
    plt.title("Distribution of Non-Hydrogen Atom Counts", fontsize=14)
    plt.xticks(freq.index[::2] if len(freq.index) > 2 else freq.index)
    plt.show()

    element_counts = Counter()
    for smi in df[smiles_col]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            elements = {atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetSymbol() not in ["C", "H"]}
            for el in elements:
                element_counts[el] += 1

    element_df = pd.DataFrame.from_dict(element_counts, orient="index", columns=["Compounds"])
    element_df = element_df.sort_values("Compounds", ascending=False)

    plt.figure(figsize=(8, 6))
    plt.bar(element_df.index, element_df["Compounds"], color="skyblue", edgecolor="k")
    plt.ylabel("Number of Compounds", fontsize=12)
    plt.xlabel("Element", fontsize=12)
    plt.title("Number of Compounds Containing Each Element (Excluding C & H)", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()

    plot_top_functional_groups_in_grid(
        df,
        smiles_col=smiles_col,
        top_n=top_n,
        max_per_row=max_per_row,
        outfile=None,
    )

    if include_reactive_group_distribution:
        df_group_count = df.copy()
        df_group_count["Reactive_Group_Count"] = df_group_count[smiles_col].apply(
            lambda s: _count_total_functional_groups(s, SMARTS_PATTERNS)
        )

        fg_freq = df_group_count["Reactive_Group_Count"].value_counts().sort_index()
        plt.figure(figsize=(8, 5))
        plt.bar(fg_freq.index, fg_freq.values, color="skyblue", edgecolor="black")
        plt.xlabel("Number of Reactive Groups per Compound")
        plt.ylabel("Number of Compounds")
        plt.title("Distribution of Reactive Group Counts per Compound")
        plt.xticks(fg_freq.index)
        plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(df[property_col], bins=property_hist_bins, color="skyblue", edgecolor="black")
    plt.xlabel(f"{property_label}{units_suffix}")
    plt.ylabel("Number of Compounds")
    plt.title(f"Distribution of {property_label} in Cleaned Dataset")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


run_bp_dataprep_plots = run_dataprep_plots
