"""
Converts a SMILES string to a 3D geometry using RDKit, then prints a Gaussian .com file to stdout.
"""

from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from tap import tapify
import re


def smiles_to_coords(smiles, cluster, CPU_IDs):
    """
    Converts a SMILES string to a 3D geometry using RDKit, then prints a Gaussian .com file to stdout.
    Single-step geometry optimization and excited-state calculation are performed using B3LYP/3-21G* (opt=(calcfc,ts,noeigen,maxcycle=1000); scrf=water; td=(singlets,nstates=5)) settings.

    :param smiles: Input SMILES string
    :param cluster: Molecule/job label for output section titles
    :param CPU_IDs: Gaussian CPU setting
    """

    print('%Mem=8GB')
    print(f'%CPU={CPU_IDs}')
    print('#p b3lyp/3-21G* opt=(calcfc,ts,noeigen,maxcycle=1000) scrf=(solvent=water) td=(singlets,nstates=5)\n')
    print(f' {cluster}_top\n')
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    mol.GetConformer()
    print('0 1')
    for i, atom in enumerate(mol.GetAtoms()):
        positions = mol.GetConformer().GetAtomPosition(i)
        print(f'{atom.GetSymbol()}    {positions.x}   {positions.y}   {positions.z}')
    print('\n')



if __name__ == "__main__":
    tapify(smiles_to_coords)


