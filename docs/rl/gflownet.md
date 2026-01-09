# Generating molecules with GFlowNet

Below are instructions for generating molecules with GFlowNet, a generative flow-based model.


## Installation

Clone the forked repository.

```bash
git clone https://github.com/swansonk14/gflownet
cd gflownet
git checkout antibiotics
```

Install dependencies.

```bash
conda create -y -n gflownet python=3.9
conda activate gflownet
pip install pandas==2.1.2
pip install chemprop==1.6.1
pip install descriptastorus==2.6.1
pip install typed-argument-parser==1.9.0
pip install -e . --find-links https://data.pyg.org/whl/torch-1.13.1+cu117.html
```

If package dependencies seem not to work, you may need to first install the exact frozen versions listed `requirements/`, i.e. `pip install -r requirements/main_3.9.txt --find-links https://data.pyg.org/whl/torch-1.13.1+cu117.html`.

**Note:** If you get the issue `ImportError: libXrender.so.1: cannot open shared object file: No such file or directory`, run `conda install -c conda-forge xorg-libxrender`.


## Generate molecules 

Run GFlowNet optimizing for S. aureus activity, solubility, SA score, and molecular weight.
```bash
python src/gflownet/tasks/seh_frag_moo.py \
    --objectives s_aureus solubility sa mw \
    --log_dir logs/s_aureus_solubility_sa_mw
```

Extract the results from the sqlite database to CSV.
```bash
python scripts/extract_results.py \
    --results_path logs/s_aureus_solubility_sa_mw/final/generated_mols_0.db \
    --save_path logs/s_aureus_solubility_sa_mw/final/molecules.csv
```

## Evaluate generated molecules

Switch to the SyntheMol conda environment for the following commands.

Compute novelty of the generated molecules.
```bash
chemfunc nearest_neighbor \
    --data_path logs/s_aureus_solubility_sa_mw/final/molecules.csv \
    --reference_data_path ../SyntheMol/rl/data/s_aureus/s_aureus_hits.csv \
    --reference_name train_hits \
    --metric tversky

chemfunc nearest_neighbor \
    --data_path logs/s_aureus_solubility_sa_mw/final/molecules.csv \
    --reference_data_path ../SyntheMol/rl/data/chembl/chembl.csv \
    --reference_name chembl \
    --metric tversky
```

Select hit molecules that satisfy novelty, diversity, and efficacy thresholds (including synthesizability and molecular weight).
```bash
python ../SyntheMol/scripts/data/select_molecules.py \
    --data_path logs/s_aureus_solubility_sa_mw/final/molecules.csv \
    --save_molecules_path logs/s_aureus_solubility_sa_mw/final/hits.csv \
    --save_analysis_path logs/s_aureus_solubility_sa_mw/final/analysis.csv \
    --score_columns "S. aureus" "Solubility" "sa_score" "molecular_weight" \
    --score_comparators ">=0.5" ">=-4" "<=4" "<=600" \
    --novelty_threshold 0.6 0.6 \
    --similarity_threshold 0.6 \
    --select_num 20 \
    --sort_column "S. aureus" \
    --descending
```

Visualize hits.
```bash
chemfunc visualize_molecules \
    --data_path logs/s_aureus_solubility_sa_mw/final/hits.csv \
    --save_dir logs/s_aureus_solubility_sa_mw/final/hits
```
