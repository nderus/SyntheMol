"""
Binarize a numeric column into zeros or ones given a threshold.
Example usage: python binarize.py data.csv PLQY 0.5
"""

import pandas as pd
import sys, os

def add_binary_column(csv_file: str, column_name: str, threshold: float) -> pd.DataFrame:
    """
    Add a binary column to a CSV file based on a threshold value.

    :param csv_file: Path to the input CSV file
    :param column_name: Name of the column to binarize
    :param threshold: Threshold value. Values greater than threshold become 1, else 0
    :return: DataFrame with an additional 'binary' column
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in CSV")

    # Add a new column 'binary' based on the condition
    df['binary'] = df[column_name].apply(lambda x: 1 if x > threshold else 0)

    return df

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python binarize.py <input_csv_file> <column_name> <threshold>")
        sys.exit(1)

    csv_file = sys.argv[1]
    column_name = sys.argv[2]

    try:
        threshold = float(sys.argv[3])
    except ValueError:
        print("Threshold must be a numeric value")
        sys.exit(1)

    result_df = add_binary_column(csv_file, column_name, threshold)

    filename, extension = os.path.splitext(csv_file)
    new_filename = f"{filename}_with_binary{extension}"

    # Save the modified DataFrame to a new CSV file
    result_df.to_csv(new_filename, index=False)

    print(f"New CSV file saved as: {new_filename}")
    print(result_df)