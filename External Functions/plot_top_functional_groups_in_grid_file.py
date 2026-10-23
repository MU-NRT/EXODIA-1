from __future__ import annotations

import pandas as pd

from plot_top_functional_groups_in_grid_v2 import plot_top_functional_groups_in_grid_v2


def plot_top_functional_groups_in_grid(
    df,
    smiles_col: str = "canonical_smiles",
    top_n: int = 20,
    max_per_row: int = 5,
    outfile: str | None = None,
    title: str = "Top Molecular Subgroups",
    enforce_exactly_one: bool = True,
    fragment_function=None,
    separate_fragment_labels: bool = False,
):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in provided DataFrame")

    smiles_list = [str(s) for s in df[smiles_col].dropna().tolist()]
    return plot_top_functional_groups_in_grid_v2(
        smiles_list=smiles_list,
        smiles_col=smiles_col,
        fragment_function=fragment_function,
        enforce_exactly_one=enforce_exactly_one,
        top_n=top_n,
        max_per_row=max_per_row,
        outfile=outfile,
        title=title,
        separate_fragment_labels=separate_fragment_labels,
    )
