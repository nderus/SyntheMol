#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_file> <output_file>"
    exit 1
fi

# Capture the input and output files
input_file="$1"
output_file="$2"

# Check if the SMILES and properties file exists
if [ ! -f "$input_file" ]; then
    echo "Error: SMILES and properties file '$smiles_and_properties_file' not found."
    exit 1
fi

# Loop through each line in the file and process SMILES and property using the Python script
while IFS=',', read -r cluster smiles; do
   python3 build_gaussian_input.py --smiles "$smiles" --cluster "$cluster" > generated_file.com
   #cat generated_file.com
   
   if [ $? -eq 0 ]; then
      echo "starting gaussian"
      g16 < generated_file.com > generated_log.log
      if [ $? -eq 0 ]; then
         echo "calculating homolumo"
         python3 calc_homolumo.py --inputfilename generated_log.log --smiles_str "$smiles">> "$output_file"
      else
         echo "error gaussian"
      fi 
   else
      echo "error input"
   fi
done < "$input_file"

echo "Processing complete."
