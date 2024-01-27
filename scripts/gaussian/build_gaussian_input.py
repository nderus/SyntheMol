from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from tap import tapify


def smiles_to_coords(smiles, cluster):
    #f = open(f'./{cluster}_top_new.com', 'w')

    print('#p b3lyp/3-21G* opt=(calcfc,ts,noeigen) scrf=(solvent=water) td=(singlets,nstates=5) scf=maxcycle=1000\n')
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


