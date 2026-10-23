def fragment_molecules_to_dataframe(df, smiles_col='canonical_smiles', enforce_exactly_one=True):
    import pandas as pd
    from rdkit import Chem

    RING_LABELS = {"aromatic", "non_aromatic_ring"}

    # ---------- Predicates ----------
    def is_aliphatic_anchor_C(atom):
        return (
            atom.GetAtomicNum() == 6
            and not atom.GetIsAromatic()
            and not atom.IsInRing()
            and all(b.GetBondType() == Chem.BondType.SINGLE for b in atom.GetBonds())
        )

    def aromatic_atom_indices(mol):
        return {a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()}

    def non_aromatic_atom_indices(mol):
        return {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing() and not a.GetIsAromatic()}

    def collect_aliphatic_anchors(mol):
        return {a.GetIdx() for a in mol.GetAtoms() if is_aliphatic_anchor_C(a)}

    def classify_fragment(atom_indices, aromatic_idxs, non_aromatic_idxs):
        if any(i in aromatic_idxs for i in atom_indices):
            return "aromatic"
        if any(i in non_aromatic_idxs for i in atom_indices):
            return "non_aromatic_ring"
        return "aliphatic"

    # ---------- Traversal ----------
    def bfs_group_blocking(mol, seed_idx, block_idxs):
        visited, stack = set(), [seed_idx]
        while stack:
            i = stack.pop()
            if i in visited or i in block_idxs:
                continue
            visited.add(i)
            ai = mol.GetAtomWithIdx(i)
            for nb in ai.GetNeighbors():
                j = nb.GetIdx()
                if j not in visited and j not in block_idxs and nb.GetAtomicNum() != 1:
                    stack.append(j)
        return tuple(sorted(visited))

    def has_adjacent_cc_multibond(mol, atom_idx, exclude_neighbor_idx):
        a = mol.GetAtomWithIdx(atom_idx)
        if a.GetAtomicNum() != 6:
            return False
        for nb in a.GetNeighbors():
            j = nb.GetIdx()
            if j == exclude_neighbor_idx or nb.GetAtomicNum() != 6:
                continue
            b = mol.GetBondBetweenAtoms(atom_idx, j)
            if b and b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE):
                return True
        return False

    def has_cc_unsat_neighbor(mol, atom_idx):
        atom = mol.GetAtomWithIdx(atom_idx)
        for nb in atom.GetNeighbors():
            if nb.GetAtomicNum() != 6:
                continue
            for nb2 in nb.GetNeighbors():
                if nb2.GetAtomicNum() != 6 or nb2.GetIdx() == atom_idx:
                    continue
                bond = mol.GetBondBetweenAtoms(nb.GetIdx(), nb2.GetIdx())
                if bond and bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE):
                    return True
        return False

    # ---------- Fragment Construction ----------
    def fragment_with_r_groups_fixed(mol, atom_indices, cut_points):
        emol = Chem.RWMol(mol)
        dummy_map = {}

        for idx in atom_indices:
            atom_keep = emol.GetAtomWithIdx(idx)
            orig_atom_keep = mol.GetAtomWithIdx(idx)

            for nb in orig_atom_keep.GetNeighbors():
                nb_idx = nb.GetIdx()
                if nb_idx in cut_points and nb_idx not in atom_indices:
                    is_cc_cut = (
                        orig_atom_keep.GetAtomicNum() == 6
                        and nb.GetAtomicNum() == 6
                    )

                    use_H = (
                        is_cc_cut
                        and has_adjacent_cc_multibond(mol, idx, nb_idx)
                        and not orig_atom_keep.IsInRing()
                    )

                    # Preserve explicit attachment points on ring atoms when anchor cuts are made.
                    # Ring cut sites should keep '*' instead of being hydrogen-capped.
                    if is_cc_cut and orig_atom_keep.IsInRing():
                        use_H = False

                    if use_H:
                        atom_keep.SetNumExplicitHs(atom_keep.GetNumExplicitHs() + 1)
                        atom_keep.SetNoImplicit(False)
                    else:
                        dummy = Chem.Atom(0)
                        dummy_idx = emol.AddAtom(dummy)
                        dummy_map[nb_idx] = dummy_idx
                        emol.AddBond(idx, dummy_idx, Chem.BondType.SINGLE)

                    if emol.GetBondBetweenAtoms(idx, nb_idx):
                        emol.RemoveBond(idx, nb_idx)

        atoms_to_use = list(atom_indices) + list(dummy_map.values())

        return Chem.MolFragmentToSmiles(
            emol,
            atomsToUse=atoms_to_use,
            canonical=True,
            allHsExplicit=False,
        )

    # ---------- Fragment Extraction ----------
    def extract_molecular_groups(mol, stats=None):
        if stats is not None:
            stats["total_processed"] += 1

        groups = set()

        anchors = collect_aliphatic_anchors(mol)
        aromatic_idxs = aromatic_atom_indices(mol)
        non_aromatic_idxs = non_aromatic_atom_indices(mol)

        cut_points = anchors

        # No anchors case
        if not cut_points:
            if stats is not None:
                stats["no_anchors"] += 1

            smi = Chem.MolToSmiles(mol, canonical=True)
            label = classify_fragment(range(mol.GetNumAtoms()), aromatic_idxs, non_aromatic_idxs)
            groups.add((smi, label))
            return sorted(groups)

        if stats is not None:
            stats["anchors_present"] += 1

        skipped_all_anchors = True
        bfs_success = False

        # Anchor processing
        for ci in anchors:
            cat = mol.GetAtomWithIdx(ci)
            nbrs = [
                n.GetIdx()
                for n in cat.GetNeighbors()
                if n.GetAtomicNum() != 1 and n.GetIdx() not in cut_points
            ]

            if enforce_exactly_one:
                if len(nbrs) == 1:
                    seeds = [nbrs[0]]
                elif len(nbrs) > 0 and has_cc_unsat_neighbor(mol, ci):
                    # Exception: allow >1 neighbors when adjacent carbon participates in C=C or C#C.
                    seeds = nbrs
                else:
                    continue
            else:
                seeds = nbrs

            skipped_all_anchors = False

            for seed in seeds:
                atoms = bfs_group_blocking(mol, seed, cut_points)
                if not atoms:
                    continue

                bfs_success = True
                smi = fragment_with_r_groups_fixed(mol, atoms, cut_points)
                label = classify_fragment(atoms, aromatic_idxs, non_aromatic_idxs)
                groups.add((smi, label))

        if not groups:
            smi = Chem.MolToSmiles(mol, canonical=True)
            label = classify_fragment(range(mol.GetNumAtoms()), aromatic_idxs, non_aromatic_idxs)
            groups.add((smi, label))

        # Post-loop diagnostics
        if stats is not None:
            if skipped_all_anchors:
                stats["all_anchors_skipped"] += 1

            if not bfs_success:
                stats["all_bfs_empty"] += 1

            stats["successful_fragmentation"] += 1

        return sorted(groups)
    def strip_ring_substituents_and_refragment(mol, ring_atom_indices, label):
        Chem.SanitizeMol(mol)
        ring_atom_indices = set(ring_atom_indices)

        substituent_fragments = []
        processed_roots = set()

        for ring_idx in ring_atom_indices:
            ring_atom = mol.GetAtomWithIdx(ring_idx)
            if not ring_atom.IsInRing():
                continue

            for nb in ring_atom.GetNeighbors():
                root_idx = nb.GetIdx()

                if not (
                    not nb.IsInRing()
                    and nb.GetAtomicNum() > 1
                    and not (
                        nb.GetAtomicNum() == 6
                        and nb.GetDegree() <= 1
                        and all(
                            n.GetAtomicNum() in (1, 6)
                            for n in nb.GetNeighbors()
                        )
                        and all(
                            mol.GetBondBetweenAtoms(nb.GetIdx(), n.GetIdx()) is not None
                            and mol.GetBondBetweenAtoms(nb.GetIdx(), n.GetIdx()).GetBondType() == Chem.BondType.SINGLE
                            for n in nb.GetNeighbors()
                        )
                    )
                ):
                    continue

                if root_idx in processed_roots:
                    continue

                processed_roots.add(root_idx)

                bond = mol.GetBondBetweenAtoms(ring_idx, root_idx)
                if bond is None:
                    continue

                bond_type = bond.GetBondType()

                # Collect substituent atoms via BFS
                visited = set()
                stack = [root_idx]
                while stack:
                    atom_idx = stack.pop()
                    if atom_idx in visited:
                        continue
                    visited.add(atom_idx)

                    atom = mol.GetAtomWithIdx(atom_idx)
                    for neigh in atom.GetNeighbors():
                        n_idx = neigh.GetIdx()
                        if n_idx not in visited and not neigh.IsInRing():
                            stack.append(n_idx)

                substituent_atoms = list(visited)

                # Mark root atom before extraction
                root_atom = mol.GetAtomWithIdx(root_idx)
                root_atom.SetAtomMapNum(999)

                sub_smi = Chem.MolFragmentToSmiles(
                    mol,
                    atomsToUse=substituent_atoms,
                    canonical=True,
                )

                root_atom.SetAtomMapNum(0)
                sub_mol = Chem.MolFromSmiles(sub_smi)
                if sub_mol is None:
                    continue

                # Find root atom in rebuilt fragment
                root_local_idx = None
                for atom in sub_mol.GetAtoms():
                    if atom.GetAtomMapNum() == 999:
                        root_local_idx = atom.GetIdx()
                        atom.SetAtomMapNum(0)
                        break

                if root_local_idx is None:
                    continue

                # Add dummy attachment
                sub_rw = Chem.RWMol(sub_mol)
                dummy_idx = sub_rw.AddAtom(Chem.Atom(0))
                sub_rw.AddBond(root_local_idx, dummy_idx, bond_type)

                sub_rw.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(sub_rw)

                sub_final = sub_rw.GetMol()
                sub_final_smi = Chem.MolToSmiles(sub_final, canonical=True)
                substituent_fragments.append((sub_final_smi, f"{label}_substituent"))

        core_mol = Chem.Mol(mol)
        Chem.Kekulize(core_mol, clearAromaticFlags=True)

        atom_rings = core_mol.GetRingInfo().AtomRings()
        if not atom_rings:
            return None, substituent_fragments

        ring_system_atoms = set()
        for ring in atom_rings:
            ring_system_atoms.update(ring)

        core_rw = Chem.RWMol(core_mol)
        dummy_indices = []
        boundary_bonds = []

        # Capture all ring-to-nonring heavy-atom bonds so the core keeps explicit attachment points.
        for bond in core_mol.GetBonds():
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()
            a_in = a in ring_system_atoms
            b_in = b in ring_system_atoms

            if a_in == b_in:
                continue

            ring_idx = a if a_in else b
            external_idx = b if a_in else a
            external_atom = core_mol.GetAtomWithIdx(external_idx)

            if external_atom.GetAtomicNum() == 1:
                continue

            boundary_bonds.append((ring_idx, external_idx, bond.GetBondType()))

        for ring_idx, external_idx, bond_type in boundary_bonds:
            if core_rw.GetBondBetweenAtoms(ring_idx, external_idx) is None:
                continue

            dummy_idx = core_rw.AddAtom(Chem.Atom(0))
            core_rw.AddBond(ring_idx, dummy_idx, bond_type)
            core_rw.RemoveBond(ring_idx, external_idx)
            dummy_indices.append(dummy_idx)

        ring_smi = Chem.MolFragmentToSmiles(
            core_rw,
            atomsToUse=list(ring_system_atoms) + dummy_indices,
            canonical=True,
        )

        return ring_smi, substituent_fragments

    # ---------- Counting Normalization ----------
    def normalize_fragment_for_counting(frag_smiles):
        return frag_smiles.replace('*', 'C')

    records_before = []
    records_after = []

    total_input = 0
    rdkit_parse_fail = 0
    no_cut_points = 0
    base_fragment_empty = 0
    ring_group_count = 0
    stripped_ring_success = 0
    final_fragment_zero = 0

    stats = {
        "total_processed": 0,
        "no_anchors": 0,
        "anchors_present": 0,
        "all_anchors_skipped": 0,
        "all_bfs_empty": 0,
        "successful_fragmentation": 0,
    }

    for parent_smi in df[smiles_col]:
        total_input += 1

        mol = Chem.MolFromSmiles(parent_smi)
        if mol is None:
            rdkit_parse_fail += 1
            continue

        # STEP 1 - Run base fragmentation only
        base_groups = extract_molecular_groups(mol, stats=stats)

        if not base_groups:
            base_fragment_empty += 1

        anchors = collect_aliphatic_anchors(mol)
        if not anchors:
            no_cut_points += 1

        for frag, label in base_groups:
            records_before.append(
                {
                    "parent_smiles": parent_smi,
                    "fragment_smiles": frag,
                    "fragment_label": label,
                    "fragment_normalized_for_counting": normalize_fragment_for_counting(frag),
                    "stage": "before_stripping",
                }
            )

        ring_groups = [(smi, label) for smi, label in base_groups if label in RING_LABELS]
        if ring_groups:
            ring_group_count += 1

        non_ring_groups = [(smi, label) for smi, label in base_groups if label not in RING_LABELS]
        final_groups = list(non_ring_groups)

        # STEP 2 - Strip substituents from ring groups
        for ring_smi, label in ring_groups:
            ring_mol = Chem.MolFromSmiles(ring_smi)
            if ring_mol is None:
                continue

            ring_smi_stripped, sub_frags = strip_ring_substituents_and_refragment(
                ring_mol,
                list(range(ring_mol.GetNumAtoms())),
                label,
            )

            if ring_smi_stripped:
                final_groups.append((ring_smi_stripped, label))
                stripped_ring_success += 1

            for frag_smi, sub_label in sub_frags:
                final_groups.append((frag_smi, sub_label))

        # STEP 3 - Write output rows
        if not final_groups:
            final_fragment_zero += 1

        for frag, label in final_groups:
            records_after.append(
                {
                    "parent_smiles": parent_smi,
                    "fragment_smiles": frag,
                    "fragment_label": label,
                    "fragment_normalized_for_counting": normalize_fragment_for_counting(frag),
                    "stage": "after_stripping",
                }
            )

    df_before = pd.DataFrame(records_before)
    df_after = pd.DataFrame(records_after)

    print("----- Fragmentation Debug Summary -----")
    print("Total input SMILES:", total_input)
    print("RDKit parse failures:", rdkit_parse_fail)
    print("Molecules with no aliphatic anchors:", no_cut_points)
    print("Base fragmentation returned empty:", base_fragment_empty)
    print("Molecules containing ring groups:", ring_group_count)
    print("Successful ring stripping events:", stripped_ring_success)
    print("Molecules with zero final fragments:", final_fragment_zero)
    print("---------------------------------------")

    print("\n----- Fragmentation Diagnostics -----")
    for k, v in stats.items():
        print(f"{k}: {v}")

    return df_before, df_after
