"""Combines Enamine REAL building block csv files with different property scores into one file with all property columns (PLQY, Emission/nm, Absorption/nm, sp2_net)."""
import pandas as pd

building_real_df = pd.read_csv('building_blocks_plqy_score_real_class.csv')
building_emi_df = pd.read_csv('building_blocks_emi_score_real.csv')
building_ab_df = pd.read_csv('building_blocks_abs_score_real.csv')
building_sp2_df = pd.read_csv('building_blocks_sp2_net_real.csv')

# Change the column titled 'binary' to 'PLQY' in one of the DataFrames
building_real_df.rename(columns={'binary': 'PLQY'}, inplace=True)

# Append 'em' column from building_emi_df to building_real_df
building_real_df['Emission/nm'] = building_emi_df['Emission/nm']

# Append 'ab' column from building_ab_df to building_real_df
building_real_df['Absorption/nm'] = building_ab_df['Absorption/nm']

# Add sp2_net from building_sp2_df to building_real_df
building_real_df['sp2_net'] = building_sp2_df['sp2_net']

# Save the updated DataFrame back to a CSV file
building_real_df.to_csv('combined_building_blocks_score_class_real_0.5.csv', index=False)
