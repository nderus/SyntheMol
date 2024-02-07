from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from tap import tapify
import re


def smiles_to_coords(smiles, cluster, CPU_IDs):
    #f = open(f'./{cluster}_top_new.com', 'w')
    cpus = re.split('-|,', CPU_IDs)
    print('%Mem=24GB')
    print(f'%CPU={CPU_IDs}')
    print(f'%GPUCPU=0,1={cpus[0]},{cpus[1]}')
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
    #return f'./{cluster}_top_new.com'



if __name__ == "__main__":
    tapify(smiles_to_coords)


