from tap import tapify

def extract_energies(lines):
    ground_state_energy = None
    excited_state_energy = None
    
    for line in lines:
        if 'Excited State' in line:
            # Extract the first excited state energy
            parts = line.split()
            excited_state_energy = float(parts[4].replace('eV', ''))

            # Convert from eV to Hartrees (1 eV = 0.0367493 Hartrees)
            excited_state_energy = parts[6]
            break
        else:
           excited_state_energy = -1


    return excited_state_energy

def calculate_emission_wavelength(excited_state_energy):
    # Planck's constant (in Joule seconds)
    h = 6.62607015e-34 

    # Speed of light (in meters per second)
    c = 299792458 

    # Energy difference (in Joules)
    delta_E = (excited_state_energy ) * 4.3597447222071e-18

    # Wavelength (in meters)
    wavelength = h * c / delta_E 

    # Convert wavelength to nanometers
    wavelength_nm = wavelength * 1e9

    return wavelength_nm

 
def main(inputfilename, smiles_str):
    f = open(inputfilename, 'r')
    lines = f.readlines()
    excited_state_energy = extract_energies(lines)
    print(smiles_str, excited_state_energy)

if __name__ == "__main__":
    tapify(main)
