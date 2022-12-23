# Steric clash checker

A tool to check steric clashes between predicted ligand pose and a target protein structure. It is build around the structure of the results provided by the blind-docking framework [DiffDock](https://github.com/gcorso/DiffDock).

## Install

Easiest way is to run the tool inside a conda environment, for this you will need to have `conda` installed.
To install the conda environment run the command below. It was tested on Apple Mac M1, for other platforms this might not work - refer to the version of the package in the `clashCheckEnv.yml` file.

```bash
git clone git@github.com:KoubaPetr/steric_clash_checker.git`
cd steric_clash_checker/
conda create --name clashCheckEnv
conda activate clashCheckEnv
```

## Example

```bash
python steric_clash_checker.py --dataset_folder_path data/user_predictions_testset --treshold 0.4
```

### How to adapt to your data

1) Inside `data/` create a directory `YOUR_DIR` for your dataset. 
2) Inside `YOUR_DIR` create a directory `TARGET_#NO` for each target protein you are interested in.
3) Inside `TARGET_#NO` put the PDB file with the structure of your target protein and all the corresponding ligand poses in the .sdf format, for which you wish to check for the steric clashes.
4) Run:

    ```bash
    python steric_clash_checker.py --dataset_folder_path data/YOUR_DIR --treshold 0.4
    ```
   
By default, we use the treshold of 0.4 Angstrom to consider an atom pair as clashing, but you can specify your own value.
See `utils/` for a script which can help with the preparation of the dataset structure.



