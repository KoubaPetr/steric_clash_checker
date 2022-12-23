import os
import shutil
import argparse

""""

Script taking the dataset output for DiffDock, discarding faulty predictions
and copying the targets from the dataset into the folders with the output poses.
This preprocessing can be then used to automatically generate the pathfiles for 
the construction of the ComplexDatasets in steric_clash_checker.py, with the aim of evaluating the clashes.

It assumes the given file structure, including the PDBBind_preprocessed folder being placed inside data/.

"""

parser = argparse.ArgumentParser()
parser.add_argument("--path_to_clean", default='../data/user_predictions_testset', type=str, help="Path to the folder with the data that should be preprocessed for the usage with the steric clash checker.")

args = parser.parse_args()

PATH_TO_CLEAN = args.path_to_clean
NUM_PREDICTIONS = 40
sub_paths_to_clean = [os.path.join(PATH_TO_CLEAN,p) for p in os.listdir(PATH_TO_CLEAN)]

for sp in sub_paths_to_clean:
    num_files = len(os.listdir(sp))
    if num_files < NUM_PREDICTIONS:
        shutil.rmtree(sp)

updated_sub_paths_to_clean = [(p,os.path.join(PATH_TO_CLEAN,p)) for p in os.listdir(PATH_TO_CLEAN)]

for name, _path in updated_sub_paths_to_clean:
    pdb_code = name.split('-')[2]
    pdb_path = os.path.join('../data/PDBBind_processed', pdb_code, pdb_code+'_protein_processed.pdb')
    shutil.copy(pdb_path,_path)