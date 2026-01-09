"""Adds a column with sp2 network size each molecule to a CSV file."""
from tap import tapify

from collections import deque
from rdkit import Chem
import sys
import pandas as pd

def updated_max_sp2_connected_atoms(mol: str) -> int:
    """Calculates the sp2 network size of a molecule.

    :param mol: SMILES string of the molecule
    :return: The sp2 network size of the molecule
    """
    mol = Chem.MolFromSmiles(mol)
    sp2_atoms_idxs = {
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2
    }
    # Store neighbors of each SP2 atom to reduce the number of method calls
    sp2_neighbors = {
        idx: [
            neighbor.GetIdx()
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()
            if neighbor.GetIdx() in sp2_atoms_idxs
        ]
        for idx in sp2_atoms_idxs
    }

    visited_global = set()
    max_count = 0

    def dfs(atom_idx: int, visited_local: set) -> int:
        visited_local.add(atom_idx)
        visited_global.add(atom_idx)

        for neighbor_idx in sp2_neighbors[atom_idx]:
            if neighbor_idx not in visited_local:
                dfs(neighbor_idx, visited_local)

        return len(visited_local)

    for atom_idx in sp2_atoms_idxs:
        if atom_idx not in visited_global:
            max_count = max(max_count, dfs(atom_idx, set()))

    return max_count

def sp2_for_file(input_file: str, smiles_column: str):
    """Adds a column with sp2 network size to a CSV file.

    :param input_file: Path to the input CSV file
    :param smiles_column: Name of the column that contains the SMILES strings
    :return: Path to the output CSV file
    """
    df = pd.read_csv(input_file)
    df['sp2_net'] = df[smiles_column].apply(updated_max_sp2_connected_atoms)
    output_file = input_file.replace(".csv", "_sp2.csv")
    df.to_csv(output_file, index=False)

    
if __name__ == "__main__":
    tapify(sp2_for_file)
