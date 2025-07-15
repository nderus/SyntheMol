from pathlib import Path

def generate_gaussian_input(xyz_path, output_path, CPU_IDs, job_name="job", solvent="Water", nstates=10):
    xyz_path = Path(xyz_path)
    output_path = Path(output_path)

    with open(xyz_path) as f:
        lines = f.readlines()

    # Extract atoms (skip first two lines of XYZ)
    atom_lines = lines[2:]
    atoms = [line.strip() for line in atom_lines if line.strip()]

    # Build input sections
    chk_name = f"{job_name}.chk"

    opt_route = f"#p B3LYP/6-31+G(d) Opt SCRF=(IEFPCM, solvent={solvent})"
    td_route  = f"#p TD(NStates={nstates}) CAM-B3LYP/6-31+G(d) SCRF=(IEFPCM, solvent={solvent}) Geom=AllCheck Guess=Read"

    # Write combined Gaussian input
    with open(output_path, "w") as f:
        f.write('%Mem=8GB\n')
        f.write(f'%CPU={CPU_IDs}\n')
        f.write(f"%chk={chk_name}\n")
        f.write(f"{opt_route}\n\n")
        f.write(f"{job_name} - geometry optimization\n\n")
        f.write("0 1\n")
        for atom in atoms:
            f.write(atom + "\n")
        f.write("\n--Link1--\n")
        f.write('%Mem=8GB\n')
        f.write(f'%CPU={CPU_IDs}\n')
        f.write(f"%chk={chk_name}\n")
        f.write(f"{td_route}\n\n")
        f.write(f"{job_name} - TD-DFT\n\n\n")

    print(f"Gaussian input written to: {output_path}")