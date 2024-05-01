from collections import deque
from rdkit import Chem
import sys
import pandas as pd

def longest_sp2_path_in_connected_components(smiles):
    """
    Find the longest path of sp2 atoms in each connected sp2 component in the molecule.
    
    Args:
    mol: RDKit Mol object representing the molecule.
    
    Returns:
    List containing the longest path length in each connected sp2 component.
    """
    mol = Chem.MolFromSmiles(smiles)
    sp2_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP2]
    sp2_bonds = [bond.GetIdx() for bond in mol.GetBonds() if bond.GetBeginAtom().GetIdx() in sp2_atoms and bond.GetEndAtom().GetIdx() in sp2_atoms]
    subgraph = Chem.PathToSubmol(mol, sp2_bonds)
    frags = Chem.GetMolFrags(subgraph, asMols=False)
    
    longest_paths = []
    for frag_atoms in frags:
        visited = set()
        max_path = 0
        for atom_idx in frag_atoms:
            if atom_idx not in visited:
                queue = deque([(atom_idx, 0)])
                visited.add(atom_idx)
                while queue:
                    current_atom, current_path_length = queue.popleft()
                    max_path = max(max_path, current_path_length)
                    neighbors = [n.GetIdx() for n in subgraph.GetAtomWithIdx(current_atom).GetNeighbors() if n.GetIdx() in frag_atoms and n.GetIdx() not in visited]
                    for neighbor in neighbors:
                        queue.append((neighbor, current_path_length + 1))
                        visited.add(neighbor)
        longest_paths.append(max_path)

    max_num = 0
    if len(longest_paths) > 0:
        max_num = max(longest_paths)
        
    
    return max_num

if __name__ == "__main__":
    file_path = ""
    df = pd.read_csv(file_path)
    df['sp2_length'] = df['SMILES'].apply(longest_sp2_path_in_connected_components)
    df.to_csv("", index=False)

