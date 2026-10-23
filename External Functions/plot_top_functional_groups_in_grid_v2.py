def plot_top_functional_groups_in_grid_v2(
    json_path=None,
    smiles_list=None,
    fragment_function=None,
    smiles_col="canonical_smiles",
    json_smiles_column="Target_SMILES",
    unique_json_by_sample_id=True,
    enforce_exactly_one=True,
    top_n=20,
    max_per_row=5,
    outfile=None,
    title="Top Molecular Subgroups",
    title_fontsize=12,
    title_fontweight="normal",
    label_fontsize=10,
    label_fontweight="normal",
    separate_fragment_labels=False,
):
    import io
    import json
    from collections import Counter, defaultdict
    from contextlib import redirect_stderr, redirect_stdout

    import pandas as pd
    from PIL import Image
    from matplotlib import pyplot as plt
    from rdkit import Chem, rdBase
    from rdkit.Chem.Draw import rdMolDraw2D

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
        }
    )

    if fragment_function is None:
        from fragment_molecules_to_dataframe_file_v3 import fragment_molecules_to_dataframe as fragment_function

    if json_path is None and smiles_list is None:
        raise ValueError("Pass at least one input source: json_path and/or smiles_list")

    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    if max_per_row < 1:
        raise ValueError("max_per_row must be >= 1")

    HALOGENS = {"F", "Cl", "Br", "I"}

    def canonicalize_smiles(smiles):
        if not isinstance(smiles, str):
            return None
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)

    def normalize_all_halogens_to_cl(frag_smiles):
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(frag_smiles)
        if mol is None:
            return None

        rw = Chem.RWMol(mol)
        for atom in rw.GetAtoms():
            if atom.GetSymbol() in HALOGENS:
                atom.SetAtomicNum(17)

        with rdBase.BlockLogs():
            norm = Chem.MolToSmiles(rw, canonical=True)
        return norm

    def normalize_fragment_for_counting(frag_smiles):
        if not isinstance(frag_smiles, str):
            return None
        return normalize_all_halogens_to_cl(frag_smiles)

    def mol_image_with_x_and_r(mol, size=(200, 200)):
        if mol is None:
            return None

        nm = Chem.Mol(mol)
        for atom in nm.GetAtoms():
            if atom.GetSymbol() in HALOGENS:
                atom.SetProp("_displayLabel", "X")
            elif atom.GetAtomicNum() == 0:
                atom.SetProp("_displayLabel", "R")

        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        opts = drawer.drawOptions()
        opts.bondLineWidth = 2.0
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, nm)
        drawer.FinishDrawing()
        return Image.open(io.BytesIO(drawer.GetDrawingText()))

    def load_smiles_from_json(path):
        with open(path, "r") as f:
            raw = json.load(f)

        df_json = pd.DataFrame(raw["data"], columns=raw["columns"])

        if json_smiles_column not in df_json.columns:
            raise ValueError(f"Column '{json_smiles_column}' not found in JSON table")

        if unique_json_by_sample_id and "Sample_ID" in df_json.columns:
            df_json = df_json.drop_duplicates(subset="Sample_ID")

        return df_json[json_smiles_column].dropna().tolist()

    smiles_pool = []

    if json_path is not None:
        smiles_pool.extend(load_smiles_from_json(json_path))

    if smiles_list is not None:
        smiles_pool.extend(smiles_list)

    total_input_smiles = len(smiles_pool)

    canonical_smiles = []
    for smi in smiles_pool:
        can = canonicalize_smiles(smi)
        if can is not None:
            canonical_smiles.append(can)

    # Deduplicate before fragmentation so each molecule contributes once.
    canonical_smiles = sorted(set(canonical_smiles))

    if not canonical_smiles:
        print("No valid SMILES available after parsing.")
        return None

    df_input = pd.DataFrame({smiles_col: canonical_smiles})

    out_buffer = io.StringIO()
    err_buffer = io.StringIO()
    with rdBase.BlockLogs(), redirect_stdout(out_buffer), redirect_stderr(err_buffer):
        try:
            _, df_frag = fragment_function(
                df_input,
                smiles_col=smiles_col,
                enforce_exactly_one=enforce_exactly_one,
            )
        except TypeError:
            # Compatibility with fragment functions that do not expose enforce_exactly_one.
            _, df_frag = fragment_function(df_input, smiles_col=smiles_col)

    if df_frag is None or df_frag.empty:
        print("Fragmentation produced no fragments.")
        return None

    required_cols = {"parent_smiles", "fragment_smiles", "fragment_label"}
    if not required_cols.issubset(df_frag.columns):
        missing_cols = required_cols.difference(df_frag.columns)
        raise ValueError(f"Fragment output missing required columns: {sorted(missing_cols)}")

    occurrence_counts = Counter()
    fragment_to_parents = defaultdict(set)
    display_smiles_by_key = {}
    labels_by_key = defaultdict(set)

    for _, row in df_frag.iterrows():
        parent_smi = row["parent_smiles"]
        frag_smiles = row["fragment_smiles"]
        frag_label = row["fragment_label"]

        norm_smiles = normalize_fragment_for_counting(frag_smiles)
        if norm_smiles is None:
            continue

        key = (norm_smiles, frag_label) if separate_fragment_labels else norm_smiles
        occurrence_counts[key] += 1
        fragment_to_parents[key].add(parent_smi)
        labels_by_key[key].add(frag_label)
        if key not in display_smiles_by_key:
            display_smiles_by_key[key] = norm_smiles

    molecule_counts = Counter({k: len(v) for k, v in fragment_to_parents.items()})

    if not molecule_counts:
        print("No valid normalized fragments available for plotting.")
        return None

    whole_molecule_groups = 0
    grouped = df_frag.groupby("parent_smiles")
    for parent_smi, group in grouped:
        uniq = set(group["fragment_smiles"].dropna().tolist())
        if len(uniq) == 1 and parent_smi in uniq:
            whole_molecule_groups += 1

    unique_fragments = {
        k[0] if isinstance(k, tuple) else k
        for k in molecule_counts.keys()
    }
    unique_fragment_label_pairs = {
        (row["fragment_label"], normalize_fragment_for_counting(row["fragment_smiles"]))
        for _, row in df_frag.iterrows()
        if normalize_fragment_for_counting(row["fragment_smiles"]) is not None
    }

    print("----- Fragment Counting Summary -----")
    print(f"SMILES strings provided: {total_input_smiles}")
    print(f"Valid unique molecules fragmented: {len(canonical_smiles)}")
    print(f"Unique normalized fragments (SMILES only): {len(unique_fragments)}")
    print(f"Unique normalized fragments (SMILES + label): {len(unique_fragment_label_pairs)}")
    print(f"Unique plotted groups: {len(molecule_counts)}")
    print(f"Molecules counted as whole (no fragmentation): {whole_molecule_groups}")
    print("------------------------------------")

    top_groups = molecule_counts.most_common(top_n)
    n_items = len(top_groups)
    n_cols = min(max_per_row, n_items)
    n_rows = (n_items + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 3.8 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes_list = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes_list = list(axes)
    else:
        axes_list = [ax for row in axes for ax in row]

    total_molecules_processed = df_frag["parent_smiles"].nunique()

    for i, (key, count) in enumerate(top_groups):
        ax = axes_list[i]
        disp_smiles = display_smiles_by_key[key]

        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(disp_smiles)

        if mol is None:
            ax.axis("off")
            continue

        img = mol_image_with_x_and_r(mol, size=(220, 220))
        ax.imshow(img)
        ax.axis("off")

        rel = 100.0 * count / total_molecules_processed if total_molecules_processed else 0.0
        group_labels = sorted(labels_by_key[key])
        if separate_fragment_labels:
            title_label = key[1]
        elif len(group_labels) == 1:
            title_label = group_labels[0]
        else:
            title_label = None

        title_lines = [f"{rel:.1f}% of molecules (n={count})"]
        if title_label:
            title_lines.insert(0, title_label)
        ax.set_title(
            "\n".join(title_lines),
            fontsize=label_fontsize,
            fontweight=label_fontweight,
        )

    for ax in axes_list[n_items:]:
        ax.axis("off")

    fig.suptitle(
        title,
        fontsize=title_fontsize,
        fontweight=title_fontweight,
    )
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=600, bbox_inches="tight")
    plt.show()

    return {
        "counts": molecule_counts,
        "fragment_table": df_frag,
        "occurrence_counts": occurrence_counts,
        "display_smiles_by_key": display_smiles_by_key,
    }
