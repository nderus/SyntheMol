# Runs the calc_homolumo.py script on existing Gaussian output .log files.
#!/bin/bash

# Define the path to the file containing the list of files
FILE_LIST="file_list.csv"

# Define the path to the Python script
PYTHON_SCRIPT="calc_homolumo.py"

# Iterate over each line in the file list
while IFS=',' read -r string file; do
    # Generate the output filename
    file=$(echo "$file" | tr -d '\r')
    output_file="${file/_log.log/_out.txt}"
    # Run the Python script on the current file and direct output to the output file
    python3 "$PYTHON_SCRIPT" --inputfilename "$file" --smiles_str "$string" > "$output_file"
done < "$FILE_LIST"
