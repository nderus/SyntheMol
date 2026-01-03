from tap import tapify
import pandas as pd

def filter_molecules(input_file: str):
    """Filter a CSV of molecules using configurable property thresholds.

    Applies the following filters sequentially: sp2, p(PLQY > 0.5), Emission/nm, Absorption/nm.
    Note in SyntheFluor emission and absorption scores in the molecules.csv from the generation are binary, 
    representing whether or not the wavelength is predicted to be in visible range (420-750nm).

    :param input_file: Path to the input molecules.csv file from generation.
    """
    df = pd.read_csv(input_file)
    print(f"The initial number of molecules is {len(df)}.")
    sp2_df = df[df['sp2'] >= 12]
    print(f"sp2 criteria filtered out {len(df) - len(sp2_df)} molecules.")
    plqy_df = sp2_df[sp2_df['PLQY'] >= 0.5]
    print(f"PLQY criteria filtered out {len(sp2_df) - len(plqy_df)} molecules.")
    emi_df = plqy_df[plqy_df['Emission/nm'] >= 0.5]
    print(f"Emission criteria filtered out {len(plqy_df) - len(emi_df)} molecules.")
    abs_df = emi_df[emi_df['Absorption/nm'] >= 0.5]
    print(f"Absorption criteria filtered out {len(emi_df) - len(abs_df)} molecules.")
    print(f"The final size is {len(abs_df)}. {len(df)-len(abs_df)} total molecules were removed.")
    output_file = input_file.replace(".csv", "_filtered.csv")
    abs_df.to_csv(output_file, index=False)

    
if __name__ == "__main__":
    tapify(filter_molecules)