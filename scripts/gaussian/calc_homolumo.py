from tap import tapify

def extract_energies(lines):
    alphaGaps = []
    excited_state_energies = []
    
    for line in lines:
        if line.find("Alpha virt. eigenvalues --") >=0:
            if prev_line.find("Alpha  occ. eigenvalues --") >= 0:
                line_removed = line.replace("Alpha virt. eigenvalues --", " ")
                prev_line_removed = prev_line.replace("Alpha  occ. eigenvalues --", " ")
                line_StateInfo = line_removed.split()
                prev_line_StateInfo = prev_line_removed.split()
                alphaGap = 1240/abs(27.211*float(prev_line_StateInfo[-1]) - float(line_StateInfo[0]))
                alphaGaps.append(alphaGap)
        prev_line = line

        if ' Excited State   1:' in line:
            parts = line.split()
            excited_state_energies.append(float(parts[6]))


    return excited_state_energies, alphaGaps

 
def main(inputfilename, smiles_str):
    f = open(inputfilename, 'r')
    lines = f.readlines()
    excited_state_energies, alphaGaps = extract_energies(lines)
    print(smiles_str, excited_state_energies, alphaGaps)

if __name__ == "__main__":
    tapify(main)
