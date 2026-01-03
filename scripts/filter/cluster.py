import pandas as pd
from tap import tapify

def create_clusters(input_file: str):
  """Generate individual CSV files for each cluster of molecules from a CSV of molecules.
  param: input_file: Path to CSV file with molecules that have been clustered with the chemfunc cluster_molcules function.
  """

  data = pd.read_csv(input_file)

  # Get unique cluster labels
  cluster_labels = data['cluster_label'].unique()

  # Iterate over each cluster label and create a CSV file for each
  for label in cluster_labels:
      # Filter data for the current cluster label
      cluster_data = data[data['cluster_label'] == label]
      index = input_file.index('molecules')
      output_dir = input_file[:index]
      cluster_data.to_csv(f'{output_dir}clusters/molecules_cluster_{label}.csv', index=False)

if __name__ == "__main__":
    tapify(create_clusters)