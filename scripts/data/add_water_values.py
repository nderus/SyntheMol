import numpy as np
import pandas as pd
from tap import tapify
def add_water_values(input_file: str):
    if not input_file.endswith('.npz'):
        raise ValueError('Input file must be a .npz file')
    features = np.load(input_file)
    #SP, SdP, SA, SB for water
    values = np.array([0.681, 0.997, 1.062, 0.025])

    append_vals = np.full((features['features'].shape[0], 4), values)
    gen_vals = np.concatenate((features['features'], append_vals), axis=1)
    output_file = input_file.replace('.npz', '_solvent.npz')
    np.savez(output_file, features=gen_vals)

if __name__ == "__main__":
    tapify(add_water_values)