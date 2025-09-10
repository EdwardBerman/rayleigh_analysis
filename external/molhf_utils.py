"""
Utility functions from https://github.com/violet-sto/MolHF, "MolHF: A Hierarchical Normalizing Flow for Molecular Graph Generation, cleaned up to include only the functions we use. 
Modified for our purposes.
"""

import re

import torch
from rdkit import Chem

num2bond = {1: Chem.rdchem.BondType.SINGLE,
            2: Chem.rdchem.BondType.DOUBLE,
            3: Chem.rdchem.BondType.TRIPLE}


DEFAULT_VALENCY = {
    7: 3,
    8: 2,
    16: 2,
}


def _check_valency(mol):
    try:
        Chem.SanitizeMol(
            mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence


def _correct_mol(mol):
    while True:
        flag, atomid_valence = _check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                queue.append(
                    (b.GetIdx(), int(b.GetBondType())-1,
                     b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                )
            queue.sort(key=lambda tup: tup[1], reverse=True)
            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                t = queue[0][1]
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, num2bond[t-1])

    return mol


def _valid_mol(mol):
    s = Chem.MolFromSmiles(Chem.MolToSmiles(
        mol, isomericSmiles=True)) if mol is not None else None
    if s is not None and '.' not in Chem.MolToSmiles(s, isomericSmiles=True):
        return s
    return None


def _valid_mol_can_with_seg(mol, largest_connected_comp=True):
    if mol is None:
        return None
    sm = Chem.MolToSmiles(mol, isomericSmiles=True)
    mol = Chem.MolFromSmiles(sm)
    if largest_connected_comp and '.' in sm:
        vsm = [(s, len(s)) for s in sm.split('.')]
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    return mol


def construct_mol(x, A, atomic_num_list):
    """Adapted implementation of construct_mol from GraphNVP."""

    # TODO: This is temporary! For ZINC, 0 is the virtual padding node
    num2atom = {1: 6, 2: 7, 3: 8, 4: 9, 5: 15, 6: 16, 7: 17, 8: 35, 9: 53}
    
    mol = Chem.RWMol()
    atoms = torch.argmax(x, axis=1)
    atoms_exist = atoms != 0
    atoms = atoms[atoms_exist]

    for atom in atoms:
        atom_idx = atom.item()
        mol.AddAtom(Chem.Atom(num2atom[atom_idx]))

    adj = A[atoms_exist, :][:, atoms_exist]
    adj[adj == 3] = -1
    adj += 1
    for start, end in torch.nonzero(adj):
        if start > end:
            mol.AddBond(start.item(), end.item(),
                        num2bond[adj[start, end].item()])

    try:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return mol, smiles
    except ValueError:
        return None
