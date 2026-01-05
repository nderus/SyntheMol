# Generating Novel Fluorescent Molecules with SyntheMol-RL
Instructions for generating fluorescent molecules using SyntheMol-RL from the paper [TODO](TODO).
This includes the instructions for processing the fluorescence data, training fluorescent property prediction models, generating molecules with SyntheMol-RL, and computational filtering methods to select candidates including instructions for how to run Gaussian. 

* [Data](#data)
    + [Process ChemFluor training datasets](#process-chemfluor-training-datasets)
* [Build fluorescent property prediction models](#build-fluorescent-property-prediction-models)
    + [Compute Morgan features](#compute-morgan-features)
    + [Compute RDKit Features](#compute-rdkit-features)
    + [Concatenate Solvent Features to Fingerprint Vector](#concatenate-solvent-features-to-fingerprint-vector)
    + [Prepare building blocks for generation](#prepare-building-blocks-for-generation)
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

## Process ChemFluor training datasets

The **ChemFluor dataset** is sourced from the following publication:

> **Cheng-Wei Ju, Hanzhi Bai, Bo Li, and Rizhang Liu**  
> *Machine learning enables highly accurate predictions of photophysical properties of organic fluorescent materials: Emission wavelengths and quantum yields*  
> **Journal of Chemical Information and Modeling**, 61(3):1053–1065, 2021  
> PMID: **33620207**  
> https://pubs.acs.org/doi/10.1021/acs.jcim.0c01295

We use the Alldata_SMILES_v0.1.xlsx file provided by the authors of this paper, which is included in the chemfluor_data section of the Data Availability folder.

- **Filename:** `Alldata_SMILES_v0.1.xlsx`
- **Description:**  
  Contains SMILES representations of organic fluorescent molecules along with experimentally measured photophysical properties, including:
  - Emission wavelength
  - Absorption wavelength
  - Photoluminescence Quantum yield


The ChemFluor dataset contains SMILES representations of organic fluorescent molecules along with their experimentally measured photophysical properties. These photophysical properties can differ depending on the solvent the molecule is in. Thus, we treat distinct molecule-solvent pairs as individual training examples; the inputs to our models contain the graph representation of the molecule, as well as numerical features encoding molecule and solvent information.

ChemFluor contains 4336 unique molecule-solvent pairs. We found that some molecule-solvent pairs were missing data for some combination of PLQY, absorption, and emission measurements, but not all. To maximize the amount of examples to train each model, we created three separate datasets for each measurement, only dropping molecule-solvent pairs that did not contain a measurement for the specified output value. We also noticed that some molecule-solvent pairs were listed multiple times with different measurements for properties (despite having consistent SMILES strings and SP/SdP/SA/SB values). Thus, we averaged measurements across these molecule-solvent pairs in each dataset.

The code for the filtering process outlined above is in the following file:

```bash
scripts/data/chemfluor_processing.ipynb
```
Since we trained classification models to predict PLQY > 0.5, we binarized the numerical PLQY data below. 

```
python scripts/data/binarize.py chemfluor_plqy_train_solvents.csv PLQY 0.5 
```

After all preprocessing, we end up with the following three training files for PLQY, emission, and absorption, respectively.  

1. chemfluor_plqy_train_solvents_with_binary.csv: 3055 molecules  
2. chemfluor_emi_train_solvents.csv: 4333 molecules  
3. chemfluor_abs_train_solvents.csv: 4202 molecules  
These files are available in Zenodo under the chemfluor_data/subfolder. 

## Build fluorescent property prediction models
We train four different property prediction architectures to predict each fluorescence property:
1. Chemprop-Morgan: a graph neural network model augmented with 2052-dimensional vector (Morgan fingerprint with solvent features)
2. Chemprop-RDKit: a graph neural network model augmented with 204-dimensional vector (RDKit fingerprint with solvent features)
3. MLP-Morgan: a multilayer perceptron using 2052-dimensional vector (Morgan fingerprint with solvent features)
4. MLP-RDKit: a multilayer perceptron using 204-dimensional vector (RDKit fingerprint with solvent features)

All were trained using the Chemprop package v1.6.1. PLQY prediction was modeled as a binary classification task using a threshold of PLQY > 0.5, while absorption and emission wavelength predictions were modeled as regression tasks.

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

This is done by extracting the SP, SdP, SA, and SB values corresponding to each training example in each dataset and concatenating them with the feature values created in the above command.

The jupyter notebook file with this can be found below:

```bash
scripts/data/concatenate_solvent_values_with_features.ipynb
```

The input and output files for this notebook can be found in the features_model_training subfolder of chemfluor_data in the Data Availability folder.

### Train models

For each model type, train 10 models using 10-fold cross-validation with an 80% training, 10% validation, and 10% testing split.
Chemprop-Morgan for PLQY

```
chemprop_train --data_path synthemol/resources/fluorescence_solvents/chemfluor_plqy_train_solvents_with_binary.csv --dataset_type classification --smiles_column canonical_smiles --target_columns binary --features_path synthemol/resources/fluorescence_solvents/chemfluor_plqy_morgan_train_features_solvent.npz --metric auc --extra_metric prc-auc accuracy f1 --save_dir synthemol/resources/models/chemprop_plqy_morg_solv_class --save_preds --split_type cv --num_folds 10 --no_features_scaling
```
Chemprop-Morgan for Emission

```
chemprop_train --data_path synthemol/resources/fluorescence_solvents/chemfluor_emi_train_solvents.csv --dataset_type regression --smiles_column canonical_smiles --target_columns Emission/nm --features_path synthemol/resources/fluorescence_solvents/chemfluor_emi_morgan_train_features_solvent.npz --metric rmse --extra_metric mae mse r2 --save_dir synthemol/resources/models/chemprop_emi_morg_solv --save_preds --split_type cv --num_folds 10 --no_features_scaling
```
Chemprop-Morgan for Absorption

```
chemprop_train --data_path synthemol/resources/fluorescence_solvents/chemfluor_abs_train_solvents.csv --dataset_type regression --smiles_column canonical_smiles --target_columns Absorption/nm --features_path synthemol/resources/fluorescence_solvents/chemfluor_abs_morgan_train_features_solvent.npz --metric rmse --extra_metric mae mse r2 --save_dir synthemol/resources/models/chemprop_abs_morg_solv --save_preds --split_type cv --num_folds 10 --no_features_scaling
```
MLP-Morgan for PLQY

```
chemprop_train --data_path synthemol/resources/fluorescence_solvents/chemfluor_plqy_train_solvents_with_binary.csv --dataset_type classification --smiles_column canonical_smiles --target_columns binary --features_path synthemol/resources/fluorescence_solvents/chemfluor_plqy_morgan_train_features_solvent.npz --metric auc --extra_metric prc-auc accuracy f1 --save_dir synthemol/resources/models/mlp_plqy_morg_solv_class --save_preds --split_type cv --num_folds 10 --no_features_scaling --features_only
```
MLP-Morgan for Emission

```
chemprop_train --data_path synthemol/resources/fluorescence_solvents/chemfluor_emi_train_solvents.csv --dataset_type regression --smiles_column canonical_smiles --target_columns Emission/nm --features_path synthemol/resources/fluorescence_solvents/chemfluor_emi_morgan_train_features_solvent.npz --metric rmse --extra_metric mae mse r2 --save_dir synthemol/resources/models/mlp_emi_morgan_solv --save_preds --split_type cv --num_folds 10 --no_features_scaling --features_only
```
MLP-Morgan for Absorption

```
chemprop_train --data_path synthemol/resources/fluorescence_solvents/chemfluor_abs_train_solvents.csv --dataset_type regression --smiles_column canonical_smiles --target_columns Absorption/nm --features_path synthemol/resources/fluorescence_solvents/chemfluor_abs_morgan_train_features_solvent.npz --metric rmse --extra_metric mae mse r2 --save_dir synthemol/resources/models/mlp_abs_morgan_solv --save_preds --split_type cv --num_folds 10 --no_features_scaling --features_only
```
Chemprop-RDKit for PLQY
```
chemprop_train --data_path synthemol/resources/fluorescence_solvents/chemfluor_plqy_train_solvents_with_binary.csv --dataset_type classification --smiles_column canonical_smiles --target_columns binary --features_path synthemol/resources/fluorescence_solvents/chemfluor_plqy_rdkit_train_features_solvent.npz --metric auc --extra_metric prc-auc accuracy f1 --save_dir synthemol/resources/models/chemprop_plqy_rdkit_solv_class --save_preds --split_type cv --num_folds 10 --no_features_scaling
```
Chemprop-RDKit for Emission
```
chemprop_train --data_path synthemol/resources/fluorescence_solvents/chemfluor_emi_train_solvents.csv --dataset_type regression --smiles_column canonical_smiles --target_columns Emission/nm --features_path synthemol/resources/fluorescence_solvents/chemfluor_emi_rdkit_train_features_solvent.npz --metric rmse --extra_metric mae mse r2 --save_dir synthemol/resources/models/chemprop_emi_rdkit_solv --save_preds --split_type cv --num_folds 10 --no_features_scaling
```
Chemprop-RDKit for Absorption
```
chemprop_train --data_path synthemol/resources/fluorescence_solvents/chemfluor_abs_train_solvents.csv  --dataset_type regression --smiles_column canonical_smiles --target_columns Absorption/nm --features_path synthemol/resources/fluorescence_solvents/chemfluor_abs_rdkit_train_features_solvent.npz --metric rmse --extra_metric mae mse r2 --save_dir synthemol/resources/models/chemprop_abs_rdkit_solv --save_preds --split_type cv --num_folds 10 --no_features_scaling
```

MLP-RDKit for PLQY
```
chemprop_train --data_path synthemol/resources/fluorescence_solvents/chemfluor_plqy_train_solvents_with_binary.csv --dataset_type classification --smiles_column canonical_smiles --target_columns binary --features_path synthemol/resources/fluorescence_solvents/chemfluor_plqy_rdkit_train_features_solvent.npz --metric auc --extra_metric prc-auc accuracy f1 --save_dir synthemol/resources/models/mlp_plqy_rdkit_solv_class --save_preds --split_type cv --num_folds 10 --no_features_scaling --features_only
```

MLP-RDKit for Emission
```
chemprop_train --data_path synthemol/resources/fluorescence_solvents/chemfluor_emi_train_solvents.csv --dataset_type regression --smiles_column canonical_smiles --target_columns Emission/nm --features_path synthemol/resources/fluorescence_solvents/chemfluor_emi_rdkit_train_features_solvent.npz --metric rmse --extra_metric mae mse r2 --save_dir synthemol/resources/models/mlp_emi_rdkit_solv --save_preds --split_type cv --num_folds 10 --no_features_scaling --features_only
```
MLP-RDKit for Absorption
```
chemprop_train --data_path synthemol/resources/fluorescence_solvents/chemfluor_abs_train_solvents.csv  --dataset_type regression --smiles_column canonical_smiles --target_columns Absorption/nm --features_path synthemol/resources/fluorescence_solvents/chemfluor_abs_rdkit_train_features_solvent.npz --metric rmse --extra_metric mae mse r2 --save_dir synthemol/resources/models/mlp_abs_rdkit_solv --save_preds --split_type cv --num_folds 10 --no_features_scaling --features_only

```

The Chemprop-Morgan architecture was selected for the reward scoring function in the SyntheFluor generation process based on its superior performance, while the MLP-architecture was selected for the RL value function due to its faster speed. To initially score the building blocks, we also used the Chemprop-Morgan architecture for consistency.

### Prepare building blocks for generation

To run these models on the building blocks, we first need to create Morgan fingerprints.
```
chemfunc save_fingerprints --fingerprint_type morgan --data_path synthemol/resources/real/real_building_blocks.csv --save_path synthemol/resources/fluorescence_solvents/real_building_blocks_features.npz --smiles_column smiles
```

Next, we concatenate the solvent constants (SP, SdP, SA, SB) for water to the feature file. We aimed to generate a compound with optimal photophysical properties in water.

```
python scripts/data/add_water_values.py real_building_blocks_features.npz
```
Lastly, we run Chemprop-Morgan on the building blocks to generate initial predictions for PLQY > 0.5, absorption, and emission.

For PLQY:
```
chemprop_predict --test_path synthemol/resources/real/real_building_blocks.csv \
                --preds_path synthemol/resources/fluorescence_solvents/building_blocks_plqy_score_real_class.csv \
                --checkpoint_dir synthemol/resources/models/chemprop_plqy_morg_solv_class \
                --features_path  synthemol/resources/fluorescence_solvents/real_building_blocks_features_solvent.npz \
                --smiles_column smiles --no_features_scaling
```
For absorption:
```
chemprop_predict --test_path synthemol/resources/real/real_building_blocks.csv \
                --preds_path synthemol/resources/fluorescence_solvents/building_blocks_abs_score_real.csv \
                --checkpoint_dir synthemol/resources/models/chemprop_abs_morg_solv \
                --features_path  synthemol/resources/fluorescence_solvents/real_building_blocks_features_solvent.npz --smiles_column smiles --no_features_scaling
```

For emission:
```
chemprop_predict --test_path synthemol/resources/real/real_building_blocks.csv \
                --preds_path synthemol/resources/fluorescence_solvents/building_blocks_emi_score_real.csv\
                --checkpoint_dir synthemol/resources/models/chemprop_emi_morg_solv \
                --features_path  synthemol/resources/fluorescence_solvents/real_building_blocks_features_solvent.npz \
                --smiles_column smiles --no_features_scaling
```

Finally, we computed sp2 network sizes for the building blocks.

```
python scripts/filter/conjugated_pi_system.py synthemol/resources/building_blocks_plqy_score_real_class.csv
```

The separate prediction files were combined into one file for generation using:

```
python synthemol/resources/combine_building_blocks.py
```

## Generate molecules with SyntheMol-RL
Generate molecules with SyntheMol-RL using dynamic property weight tuning with the following success thresholds: 
1. PLQY: the probability of PLQY >0.5 (the classification threshold) is at least 0.5 (i.e.,  (PLQY > 0.5) ≥0.5). 
2. Absorption: the predicted wavelength is within 420 nm to 750 nm (i.e. visible spectrum). 
3. Emission: the predicted wavelength is within 420 nm to 750 nm (i.e. visible spectrum). 
4. sp2: the largest sp2 network is ≥12 atoms.

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
We create 100 clusters using k-means clustering:

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

## Gaussian

### Prerequisites
Before starting the Gaussian analysis, ensure that Gaussian 16 is installed and properly configured on your system. We used Slurm as the workload manager for job scheduling and resource allocation. Each Gaussian Job was allocated 12 CPU cores and 48 GB of memory.

### Generating Input File
The `gaussian_test.sh` script processes one molecule per run. Each input file should be a plain text file containing a single line with the format: `<molecule_id>` <smiles_string>. For example: `1 CN=C=O`. 

`molecule_id` can be any unique identifier for the molecule. In our workflow, we used the cluster number assigned during the clustering step. Before running the `gaussian_test.sh` script, modify the `input_file` and `output_file` variables in `gaussian_test.sh` to reflect your input and output file names.  

### Starting the Gaussian Job
The following instructions assume you are using Slurm as your workload manager. If you use a different workload manager, you will need to modify the scripts accordingly. 

To initiate the Gaussian workflow, run `sbatch gaussian_test.sh`. This script performs three main tasks: it generates the `.com` file required by Gaussian, submits the Gaussian job, and parses the excited state energy, oscillator strength, and dipole moment from the resulting `.log` file. 

In our analysis, the previous filtering steps retained 34 candidate molecules from the original 11,590 molecules generated by SyntheFluor. The Gaussian filtering step further reduced this set by eliminating 15 molecules with oscillator strengths below 0.1, leaving 19 final candidates. 
