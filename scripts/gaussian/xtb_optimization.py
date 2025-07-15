from rdkit import Chem
from rdkit.Chem import AllChem
import tblite.interface as tb
import numpy as np
from berny import Berny, geomlib, angstrom
import os
from tap import tapify
from pathlib import Path
from gaussian_utils import generate_gaussian_input


def write_initial_xyz_file(smiles, in_path):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    mol.GetConformer()
    with open(in_path, "w") as f:
        num_atoms = mol.GetNumAtoms()
        f.write(f"{num_atoms}\n")
        f.write("Generated from SMILES\n")
        for i, atom in enumerate(mol.GetAtoms()):
            positions = mol.GetConformer().GetAtomPosition(i)
            f.write(f'{atom.GetSymbol()}    {positions.x}   {positions.y}   {positions.z}\n')
        f.write('\n')

def optimize_molecule(xyz_path):
    # initialize optimizer and grab initial geometry
    optimizer = Berny(geomlib.readfile(xyz_path))
    geom = next(optimizer)
    elements = [symbol for symbol, _ in geom]
    initial_coordinates = np.asarray([coordinate for _, coordinate in geom])

    # set up caluclation
    xtb = tb.Calculator("GFN2-xTB", tb.symbols_to_numbers(elements), initial_coordinates * angstrom)
    results = xtb.singlepoint()
    initial_energy = results["energy"]
    initial_gradient = results["gradient"]
    
    optimizer.send((initial_energy, initial_gradient / angstrom))
    trajectory = [(initial_energy, initial_gradient, initial_coordinates)]

    # do the optimization
    xtb.set("verbosity", 0)
    for geom in optimizer:
        coordinates = np.asarray([coordinate for _, coordinate in geom])
        xtb.update(positions=coordinates * angstrom)
        results = xtb.singlepoint(results)
    
        energy = results["energy"]
        gradient = results["gradient"]
        optimizer.send((energy, gradient / angstrom))
    
        trajectory.append((energy, gradient, coordinates))
    return trajectory, elements


def write_optimized_xyz_file(trajectory, elements, output_path):
    final_energy, final_gradient, final_coordinates = trajectory[-1]
    
    # Create XYZ lines
    xyz_lines = [
        str(len(elements)),
        f"Optimized structure, Energy = {final_energy:.6f} Hartree"
    ]
    for symbol, (x, y, z) in zip(elements, final_coordinates):
        xyz_lines.append(f"{symbol} {x:.8f} {y:.8f} {z:.8f}")
    
    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(xyz_lines))
    
    print(f"Optimized structure saved to {output_path}")


 
def generate_optimal_xyz(id, smiles, input_xyz_path, cpu_ids=""):
    write_initial_xyz_file(smiles, input_xyz_path)
    directory = os.path.dirname(input_xyz_path)
    filename = os.path.basename(input_xyz_path)
    output_path = Path(directory) / f"optimized_{filename}"
    trajectory, elements = optimize_molecule(input_xyz_path)
    write_optimized_xyz_file(trajectory, elements, output_path)
    generate_gaussian_input(output_path, f"gaussian_{id}.com", CPU_IDs = cpu_ids, job_name=f'fluor_{id}')
    
    return trajectory, elements


if __name__ == "__main__":
    tapify(generate_optimal_xyz)