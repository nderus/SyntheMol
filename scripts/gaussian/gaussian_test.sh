#! /bin/bash
#
#SBATCH --job-name=gaussian
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --array=0
#SBATCH --partition=gpu
#SBATCH --gres gpu:1
#SBATCH --output=output/synthemol/%a.out
#SBATCH --error=error/synthemol/%a.err
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jennxu23@stanford.edu


source /home/users/jennxu23/.bashrc
ml load python/3.9.0
ml load chemistry gaussian
ml load py-pandas/2.0.1_py39
cd /home/users/jennxu23/synthemol/SyntheMol_private/scripts/gaussian

# Capture the input and output files
input_file="sumita.txt"
output_file="sumita_out.log"

# Check if the SMILES and properties file exists
if [ ! -f "$input_file" ]; then
    echo "Error: SMILES and properties file ${input_file}  not found."
    exit 1
fi

   
   
while IFS=',', read -r cluster smiles; do
   python3 build_gaussian_input.py --smiles "${smiles}" --cluster "${cluster}" > generated_file.com
   
   if [ $? -eq 0 ]; then
      echo "starting gaussian"
      g16 < generated_file.com
      #python3 calc_homolumo.py --inputfilename generated_log.log --smiles_str "$smiles"> "$output_file"
   fi
done < "$input_file"

echo "Processing complete."
