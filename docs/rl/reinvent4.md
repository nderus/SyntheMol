# Generating molecules with REINVENT 4

Below are instructions for generating molecules with REINVENT 4, a generative flow-based model.


## Installation

Clone the forked repository.

```bash
git clone https://github.com/swansonk14/REINVENT4
cd REINVENT4
git checkout antibiotics
```

Install dependencies.

```bash
conda create -y -n reinvent4 python=3.10
conda activate reinvent4
pip install pandas==2.1.2
pip install chemprop==1.6.1
pip install descriptastorus==2.6.1
pip install typed-argument-parser==1.9.0
python install.py cu124
```

**Note:** If you get the issue `ImportError: libXrender.so.1: cannot open shared object file: No such file or directory`, run `conda install -c conda-forge xorg-libxrender`.

## Download prior

Download the reinvent prior.

```bash
wget https://zenodo.org/records/15641297/files/reinvent.prior
mv reinvent.prior priors
```

## Generate molecules

```bash
mkdir -p runs/antibiotics
cd runs/antibiotics
for WEIGHT in 1 2 5 10 100
do
reinvent -l antibiotics.log ../../configs/antibiotics/antibiotics_weight_${WEIGHT}.toml
done
cd ../..
```

## Filter to valid molecules

```bash
for WEIGHT in 1 2 5 10 100
do
python -c "import pandas as pd
data = pd.read_csv('runs/antibiotics_weight_${WEIGHT}/staged_learning_1.csv')
data = data[data['SMILES_state'] == 1]
data.rename(columns={'SMILES': 'smiles'}, inplace=True)
data.to_csv('runs/antibiotics_weight_${WEIGHT}/molecules.csv', index=False)"
done
```

## Evaluate generated molecules

Switch to the SyntheMol conda environment for the following commands.

Compute novelty of the generated molecules.
```bash
for WEIGHT in 1 2 5 10 100
do
chemfunc nearest_neighbor \
    --data_path runs/antibiotics_weight_${WEIGHT}/molecules.csv \
    --reference_data_path ../SyntheMol/rl/data/s_aureus/s_aureus_hits.csv \
    --reference_name train_hits \
    --metric tversky

chemfunc nearest_neighbor \
    --data_path runs/antibiotics_weight_${WEIGHT}/molecules.csv \
    --reference_data_path ../SyntheMol/rl/data/chembl/chembl.csv \
    --reference_name chembl \
    --metric tversky
done
```

Select hit molecules that satisfy novelty, diversity, and efficacy thresholds (including synthesizability and molecular weight).
```bash
for WEIGHT in 1 2 5 10 100
do
python ../SyntheMol/scripts/data/select_molecules.py \
    --data_path runs/antibiotics_weight_${WEIGHT}/molecules.csv \
    --save_molecules_path runs/antibiotics_weight_${WEIGHT}/hits.csv \
    --save_analysis_path runs/antibiotics_weight_${WEIGHT}/analysis.csv \
    --score_columns "Antibiotic (raw)" "Solubility (raw)" "SAScore (raw)" "MolecularWeight (raw)" \
    --score_comparators ">=0.5" ">=-4" "<=4" "<=600" \
    --novelty_threshold 0.6 0.6 \
    --similarity_threshold 0.6 \
    --select_num 20 \
    --sort_column "Antibiotic" \
    --descending
done
```

Visualize hits.
```bash
for WEIGHT in 1 2 5 10 100
do
chemfunc visualize_molecules \
    --data_path runs/antibiotics_weight_${WEIGHT}/hits.csv \
    --save_dir runs/antibiotics_weight_${WEIGHT}/hits
done
```

Since antibiotic weight 5 leads to the highest number of hits, we use hits from that run.
