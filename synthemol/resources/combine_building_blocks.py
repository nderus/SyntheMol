import pandas as pd

building_real_df = pd.read_csv('building_blocks_plqy_score_real_class.csv')
building_emi_df = pd.read_csv('building_blocks_emi_score_real.csv')
building_ab_df = pd.read_csv('building_blocks_abs_score_real.csv')

smiles_equal = (building_real_df['smiles'] == building_emi_df['smiles']).all() and (building_real_df['smiles'] == building_ab_df['smiles']).all()
size_equal = len(building_real_df) == len(building_emi_df) == len(building_ab_df)

if smiles_equal and size_equal:
    print("The 'smiles' column is in the same order and size in all three DataFrames.")
else:
    print("The 'smiles' column is not in the same order and size in all three DataFrames.")

# Change the column titled 'binary' to 'PLQY' in one of the DataFrames
building_real_df.rename(columns={'binary': 'PLQY'}, inplace=True)

# Append 'em' column from building_emi_df to building_real_df
building_real_df['Emission/nm'] = building_emi_df['Emission/nm']

# Append 'ab' column from building_ab_df to building_real_df
building_real_df['Absorption/nm'] = building_ab_df['Absorption/nm']

# Save the updated DataFrame back to a CSV file
building_real_df.to_csv('combined_building_blocks_score_class_real_0.5.csv', index=False)
