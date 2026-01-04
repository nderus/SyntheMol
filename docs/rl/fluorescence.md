# Generating Novel Fluorescent Molecules with SyntheMol-RL
Instructions for generating fluorescent molecules using SyntheMol-RL from the paper [TODO](TODO).
This includes the instructions for processing the fluorescence data, training fluorescent property prediction models, generating molecules with SyntheMol-RL, and computational filtering methods to select candidates including instructions for how to run Gaussian. 

* [Data](#data)
    + [Process Chemfluor training datasets](#process-chemfluor-training-datasets)
* [Build fluorescent property prediction models](#build-fluorescent-property-prediction-models)
    + [Compute Morgan features](#compute-morgan-features)
    + [Compute RDKit Features](#compute-rdkit-features)
    + [Concatenate Solvent Features to Fingerprint Vector](#concatenate-solvent-features-to-fingerprint-vector)
    + [Compute model scores for building blocks](#compute-model-scores-for-building-blocks)
* [Generate molecules with SyntheMol-RL](#generate-molecules-with-synthemol-rl)
* [Select Generated Molecules](#select-generated-molecules)
    + [Property Filtering](#property-filtering)
    + [Clustering](#clustering)
    + [Visualize Selected Molecules](#visualize-selected-molecules)
* [Gaussian](#gaussian)
    + [Prerequisites](#prerequisites)
    + [Generating Input File](#generating-input-file)
    + [Running Gaussian script](#running-gaussian-script)


## Data
### Process Chemfluor training datasets [TODO: finish section including adding files to Zenodo]
ChemFluor dataset available here: https://figshare.com/articles/dataset/ChemFluor/12110619/3?file=24007736
File is Alldata_SMILES_v0.1.xlsx

```bash
python scripts/data/chemfluor_processing.ipynb
```
To binarize PLQY file,
 
```
python binarize.py chemfluor_plqy_train_solvents.csv PLQY 0.5
```

TODO: demonstrate Chemfluor filtering steps from raw Chemflour dataset csv download. We need to have it for each of the following files that are referenced below in the training models:
chemfluor_plqy_train_solvents_with_binary.csv: 3055 molecules
chemfluor_emi_train_solvents.csv: 4333 molecules
chemfluor_abs_train_solvents.csv: 4202 molecules
The results of scripts used for the SyntheFluor generation are available in Zenodo under the chemfluor_data/ subfolder. 

## Build fluorescent property prediction models
We train four different property prediction architectures to predict each fluorescence property:
Chemprop-Morgan: a graph neural network model augmented with 2052-dimensional vector (Morgan fingerprint with solvent features)
Chemprop-RDKit: a graph neural network model augmented with 204-dimensional vector (RDKit fingerprint with solvent features)
MLP-Morgan: a multilayer perceptron using 2052-dimensional vector (Morgan fingerprint with solvent features)
MLP-RDKit: a multilayer perceptron using 204-dimensional vector (RDKit fingerprint with solvent features)
All were trained using the Chemprop package v1.6.1. PLQY prediction was modeled as a binary classification task using a threshold of PLQY > 0.5, while
absorption and emission wavelength predictions were modeled as regression tasks.
### Compute Morgan features
PLQY Training Data

```
chemfunc save_fingerprints --fingerprint_type morgan --data_path synthemol/resources/fluorescence/chemfluor_plqy_train_solvents_with_binary.csv --save_path synthemol/resources/fluorescence/chemfluor_morgan_train_features.npz --smiles_column SMILES
```
Emission Training Data

```
chemfunc save_fingerprints --fingerprint_type morgan --data_path synthemol/resources/fluorescence_solvents/chemfluor_emi_train_solvents.csv --save_path synthemol/resources/fluorescence/chemfluor_morgan_train_features.npz --smiles_column SMILES
```
Absorption Training Data

```
chemfunc save_fingerprints --fingerprint_type morgan --data_path synthemol/resources/fluorescence_solvents/chemfluor_emi_train_solvents.csv --save_path synthemol/resources/fluorescence/chemfluor_morgan_train_features.npz --smiles_column SMILES
```
Building Blocks

```
chemfunc save_fingerprints --fingerprint_type morgan --data_path synthemol/resources/real/building_blocks_real.csv --save_path synthemol/resources/fluorescence/building_blocks_real_morgan_features.npz --smiles_column smiles
```
### Compute RDKit Features
PLQY Training Data

```
chemfunc save_fingerprints --fingerprint_type rdkit --data_path synthemol/resources/fluorescence/chemfluor_plqy_train_solvents_with_binary.csv --save_path synthemol/resources/fluorescence/chemfluor_rdkit_train_features.npz --smiles_column SMILES
```
Emission Training Data

```
chemfunc save_fingerprints --fingerprint_type morgan --data_path synthemol/resources/fluorescence_solvents/chemfluor_emi_train_solvents.csv --save_path synthemol/resources/fluorescence/chemfluor_rdkit_train_features.npz --smiles_column SMILES
```
Absorption Training Data

```
chemfunc save_fingerprints --fingerprint_type morgan --data_path synthemol/resources/fluorescence_solvents/chemfluor_abs_train_solvents.csv --save_path synthemol/resources/fluorescence/chemfluor_rdkit_train_features.npz --smiles_column SMILES
```
### Concatenate Solvent Features to Fingerprint Vector
[TODO: add script for this]
### Train models
TODO: add the commands for Chemprop-RDKit and MLP-RDKit
For each model type, train 10 models using 10-fold cross-validation with an 80% training, 10% validation, and 10% testing split.
Chemprop-Morgan for PLQY

```
python /home/users/rsayana/miniconda3/envs/synthemol/bin/chemprop_train --data_path synthemol/resources/fluorescence_solvents/chemfluor_plqy_train_solvents_with_binary.csv --dataset_type classification --smiles_column canonical_smiles --target_columns binary --features_path synthemol/resources/fluorescence_solvents/chemfluor_plqy_morgan_train_features_solvent.npz --metric auc --extra_metric prc-auc accuracy f1 --save_dir synthemol/resources/models/chemprop_plqy_morg_solv_class --save_preds --split_type cv --num_folds 10 --no_features_scaling
```
Chemprop-Morgan for Emission

```
python /home/users/rsayana/miniconda3/envs/synthemol/bin/chemprop_train --data_path synthemol/resources/fluorescence_solvents/chemfluor_emi_train_solvents.csv --dataset_type regression --smiles_column canonical_smiles --target_columns Emission/nm --features_path synthemol/resources/fluorescence_solvents/chemfluor_emi_morgan_train_features_solvent.npz --metric rmse --extra_metric mae mse r2 --save_dir synthemol/resources/models/chemprop_emi_morg_solv --save_preds --split_type cv --num_folds 10 --no_features_scaling
```
Chemprop-Morgan for Absorption

```
python /home/users/rsayana/miniconda3/envs/synthemol/bin/chemprop_train --data_path synthemol/resources/fluorescence_solvents/chemfluor_abs_train_solvents.csv --dataset_type regression --smiles_column canonical_smiles --target_columns Absorption/nm --features_path synthemol/resources/fluorescence_solvents/chemfluor_abs_morgan_train_features_solvent.npz --metric rmse --extra_metric mae mse r2 --save_dir synthemol/resources/models/chemprop_abs_morg_solv --save_preds --split_type cv --num_folds 10 --no_features_scaling
```
MLP-Morgan for PLQY

```
python /home/users/rsayana/miniconda3/envs/synthemol/bin/chemprop_train --data_path synthemol/resources/fluorescence_solvents/chemfluor_plqy_train_solvents_with_binary.csv --dataset_type classification --smiles_column canonical_smiles --target_columns binary --features_path synthemol/resources/fluorescence_solvents/chemfluor_plqy_morgan_train_features_solvent.npz --metric auc --extra_metric prc-auc accuracy f1 --save_dir synthemol/resources/models/mlp_plqy_morg_solv_class --save_preds --split_type cv --num_folds 10 --no_features_scaling --features_only
```
MLP-Morgan for Emission

```
python /home/users/rsayana/miniconda3/envs/synthemol/bin/chemprop_train --data_path synthemol/resources/fluorescence_solvents/chemfluor_emi_train_solvents.csv --dataset_type regression --smiles_column canonical_smiles --target_columns Emission/nm --features_path synthemol/resources/fluorescence_solvents/chemfluor_emi_morgan_train_features_solvent.npz --metric rmse --extra_metric mae mse r2 --save_dir synthemol/resources/models/mlp_emi_morgan_solv --save_preds --split_type cv --num_folds 10 --no_features_scaling --features_only
```
MLP-Morgan for Absoprtion

```
python /home/users/rsayana/miniconda3/envs/synthemol/bin/chemprop_train --data_path synthemol/resources/fluorescence_solvents/chemfluor_abs_train_solvents.csv --dataset_type regression --smiles_column canonical_smiles --target_columns Absorption/nm --features_path synthemol/resources/fluorescence_solvents/chemfluor_abs_morgan_train_features_solvent.npz --metric rmse --extra_metric mae mse r2 --save_dir synthemol/resources/models/mlp_abs_morgan_solv --save_preds --split_type cv --num_folds 10 --no_features_scaling --features_only
```
Results for PLQY
[TODO: add column with roc-auc and time]
Results for Emission
[TODO: add column with Mae and time]
Results for Absorption
[TODO: add column with Mae and time]
The Chemprop-Morgan architecture was selected for the reward scoring function in the SyntheFluor generation process based on its superior performance, while the MLP-architecture was selected for the RL value function due to its faster speed. [TODO: say which model is used for building blocks]

### Compute model scores for building blocks
[TODO: add scripts to run chemprop_predict for fingerprints]

## Generate molecules with SyntheMol-RL
Generate molecules with SyntheMol-RL using dynamic property weight tuning with the following success thresholds: 
PLQY: the probability of PLQY >0.5 (the classification threshold) is at least 0.5 (i.e., p(PLQY > 0.5) ≥0.5). 
Absorption: the predicted wavelength is within 420 nm to 750 nm (i.e. visible spectrum). 
Emission: the predicted wavelength is within 420 nm to 750 nm (i.e. visible spectrum). 
sp2: the largest sp2 network is ≥12 atoms.

```bash
synthemol \
 --score_model_paths synthemol/resources/models/chemprop_plqy_morg_solv_class_0.5 synthemol/resources/models/chemprop_emi_morg_solv synthemol/resources/models/chemprop_abs_morg_solv None\
 --score_types chemprop wavelength wavelength sp2_network\
 --score_fingerprint_types morgan morgan morgan None\
 --score_names PLQY Emission/nm Absorption/nm sp2\
 --save_dir generations/synthefluor_generation \
 --building_blocks_paths synthemol/resources/combined_building_blocks_score_class_real_0.5_sp2.csv \
  --building_blocks_score_column PLQY Emission/nm Absorption/nm sp2_net\
  --n_rollout 10000 \
  --success_thresholds ">=0.5" ">=0.5" ">=0.5" ">=12"\
  --search_type rl \
  --rl_model_type mlp  \
  --rl_model_fingerprint_type morgan \
  --rl_model_paths synthemol/resources/models/mlp_plqy_morg_solv_class_0.5/fold_0/model_0/model.pt synthemol/resources/models/mlp_emi_morgan_solv/fold_0/model_0/model.pt synthemol/resources/models/mlp_abs_morgan_solv/fold_0/model_0/model.pt None\
  --chemical_spaces real \
  --reaction_to_building_blocks_paths synthemol/resources/real/reaction_to_building_blocks_expanded.pkl \
  --h2o_solvents \
  --rl_prediction_type classification regression regression regression\
  --wandb_project_name fluorescence \
  --wandb_run_name synthefluor_generation \
  --wandb_log \
  --use_gpu
```
## Select Generated Molecules
### Property Filtering
To select the most promising candidates, use a multi-step property filtering process for generated molecules. First, remove molecules with an sp2 network size smaller than 12. Next, only retain molecules with p(PLQY >0.5) ≥ 0.5. Finally, remove molecules with a predicted absorption and emission wavelengths outside
the visible range (420–750 nm). The script below performs this multi-step property filtering:

```
python scripts/filter/property_filter.py --input_file generations/synthefluor_generation/molecules.csv
```
```
Output:
The initial number of molecules is 11590.
sp2 criteria filtered out 5479 molecules.
PLQY criteria filtered out 4256 molecules.
Emission criteria filtered out 21 molecules.
Absorption criteria filtered out 1203 molecules.
The final size is 631. 10959 total molecules were removed.
```

### Clustering
Cluster molecules:

```bash
chemfunc cluster_molecules --save_path ../../generations/synthefluor_generation/molecules_filtered_cluster100.csv --num_cluster 100 --smiles_column smiles --data_path ../../generations/synthefluor_generation/molecules_filtered.csv
```
Create csv for each cluster:

```
python cluster.py --input_file ../../generations/synthefluor_generation/molecules_filtered_cluster100.csv
```
For every generated molecule in every cluster, add tanimoto similarity score to most similar fluorescent molecule in chemfluor dataset:

```bash
for i in {1..100}; do chemfunc nearest_neighbor --data_path generations/synthefluor_generation/clusters/molecules_cluster_$i.csv --reference_data_path synthemol/resources/fluorescence/chemfluor_positives.csv --reference_name known_fluorescent --metric tanimoto --reference_smiles_column SMILES --save_path generations/synthefluor_generation/clusters_w_similarity/molecules_cluster_$i.csv; done
```
### Visualize Selected Molecules 
For every generated molecule in every cluster, create an image of the molecular structure. 

```bash
for i in {1..100}; do chemfunc visualize_molecules --data_path generations/synthefluor_generation/clusters/molecules_cluster_$i.csv --save_dir generations/synthefluor_generation/viz/cluster$i; done
```

## Gaussian [TODO: complete section]
