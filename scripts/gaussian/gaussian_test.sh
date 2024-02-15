#! /bin/bash
#
#SBATCH --job-name=gaussian
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --array=0
#SBATCH --partition=gpu,owners
#SBATCH --gres gpu:1
#SBATCH --output=output/synthemol/%a.out
#SBATCH --error=error/synthemol/%a.err
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jennxu23@stanford.edu
#SBATCH --constraint=GPU_MEM:32GB

source /home/users/jennxu23/.bashrc
ml load python/3.9.0
ml load chemistry gaussian
ml load py-pandas/2.0.1_py39
cd /scratch/users/jennxu23/SyntheMol_private/scripts/gaussian

# Capture the input and output files
input_file="fluorescence.txt"
output_file="fluorescence_out.txt"

# Check if the SMILES and properties file exists
if [ ! -f "$input_file" ]; then
    echo "Error: SMILES and properties file ${input_file}  not found."
    exit 1
fi

   
CPUS=$(taskset -cp $$ | awk -F':' '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')
   
read -r cluster smiles < "$input_file"
python3 build_gaussian_input.py --smiles "${smiles}" --cluster "${cluster}" --CPU_IDs "${CPUS}"> generated_file.com
   
if [ $? -eq 0 ]; then
   echo "starting gaussian: {$smiles}"
   g16 < generated_file.com > "${cluster}_log.log"
   python3 calc_homolumo.py --inputfilename "${cluster}_log.log" --smiles_str "$smiles">> "$output_file"
   echo "gaussian complete"
fi

done 

echo "Processing complete."
