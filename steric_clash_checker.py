import Bio.PDB
import numpy as np
import logging
import os
import argparse
from rdkit.Chem import PandasTools
from rdkit.Chem.rdchem import Conformer
from pandas import DataFrame as df
from sklearn.neighbors import KDTree as KDTree
from dataclasses import dataclass
from typing import Tuple, List, Dict
from tqdm import tqdm

"""

TODO: Describe this file 

"""

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_folder_path", default='example_files/user_predictions_testset', type=str, help="Path to the folder containing the subfolders holding the target PDB file and corresponding ligand pose .sdf files.")
parser.add_argument("--paths_file", default=None, type=str, help="Path to the file containing the target-ligand path pairs.")
parser.add_argument("--treshold", default=0.4, type=float, help="Max distance between target-ligand atoms in Angstrom, so they are considered as clashing.")

@dataclass
class Ligand:
    parsed_df: df
    name: str = None
    num_heavy_atoms: int = None
    atom_coords: np.ndarray = None

    def __post_init__(self) -> None:
        try:
            self.num_heavy_atoms = self.parsed_df['ROMol'][0].GetNumHeavyAtoms()
            conformer: Conformer = self.parsed_df['ROMol'][0].GetConformer()
            self.atom_coords = np.array([conformer.GetAtomPosition(a_idx) for a_idx in range(self.num_heavy_atoms)])
        except:
            logging.warning("SKIPPING ligand pose: Encountered ligand pose with irregularity in the parsed dataframe.")

class Target:
    def __init__(self, path: str, name: str, silence_pdb_parser: bool = True):
        self.path = path
        self.name = name
        pdb_parser = Bio.PDB.PDBParser(QUIET=silence_pdb_parser)
        self.target_structure = pdb_parser.get_structure(self.name, self.path)
        self.residues = None
        self.heavy_atom_coords = []
        self.hydrogen_coords = []
        self.get_residues()
        self.get_atoms()
        self.heavy_atom_kdtree = None
        self.hydrogen_kdtree = None

    def get_residues(self) -> None:
        """
        Extract the residues or throw an exception in case of a differently organized PDB file

        :return:
        """
        try:
            self.residues = [t for t in self.target_structure[0]][-1]
        except:
            raise ValueError("Probably different format of the target protein PDB")

    def get_atoms(self) -> None:
        logging.debug('Reading atoms for the target protein')
        for residue in self.residues:
            for atom in residue:
                atom_xyz = atom.coord
                is_heavy = False if atom.name == 'H' else True # The separtion is strictly into 'H' and heavy atoms, right?
                if is_heavy:
                    self.heavy_atom_coords.append(atom_xyz) #TODO: maybe keep the reference to the atom?
                else:
                    self.hydrogen_coords.append(atom_xyz)
        logging.debug('Finished reading atoms for the target protein')

    def build_kdtrees(self) -> None:
        """
        Building KDTrees for efficient nearest neighbor search, useful for checking the clashes
        :return:
        """
        logging.debug("--- Building heavy atom KDTree ---")
        self.heavy_atom_kdtree = KDTree(np.array(self.heavy_atom_coords))
        logging.debug("--- Finished building heavy atom KDTree ---")
        logging.debug("--- Building hydrogen atoms KDTree ---")
        self.hydrogen_kdtree = KDTree(np.array(self.hydrogen_coords))
        logging.debug("--- Finished building hydrogen atom KDTree ---")

@dataclass
class Complex:
    target: Target
    ligand: Ligand
    hydrogen_nn_dists: np.ndarray = None
    hydrogen_nn_idcs: np.ndarray = None
    heavy_nn_dists: np.ndarray = None
    heavy_nn_idcs: np.ndarray = None
    heavy_atom_clash: bool = None
    hydrogen_clash: bool = None
    last_clash_treshold: float = None

    def __post_init__(self):
        """
        To compute the nearest neighbors upon instantiation
        :return:
        """
        self.hydrogen_nn_dists, self.hydrogen_nn_idcs = self.target.hydrogen_kdtree.query(self.ligand.atom_coords)
        self.heavy_nn_dists, self.heavy_nn_idcs = self.target.heavy_atom_kdtree.query(self.ligand.atom_coords)

    def check_clashes(self, treshold: float = 0.4) -> Tuple[bool,bool]:
        """
        Function to check steric clashes given a treshold on a minimum allowed distance 
        of a ligand atom (only heavy atoms for now) and target atoms (both heavy and hydrogen atoms).

        :param treshold: in Angstroms, minimum allowed distance for non-clashing atoms,
                         default value = 0.4A (same as DiffDock paper)
        :return:  Tuple[bool, bool] - bool values for hydrogen/heavy atoms clashes
        """
        self.hydrogen_clash = self.hydrogen_nn_dists.min() < treshold
        self.heavy_atom_clash = self.heavy_nn_dists.min() < treshold
        self.last_clash_treshold = treshold
        return self.hydrogen_clash, self.heavy_atom_clash


class ComplexDataset:

    def __init__(self, path_pairs: List[Tuple[str,str]] = None, pathsfile_path: str = None, clash_treshold: float = None):
        self.target_ligand_path_pairs: List[Tuple[str,str]] = path_pairs
        self.pathsfile_path: str = pathsfile_path
        self.clash_treshold:float = clash_treshold

        self.complexes: List[Complex] = []
        self.hydrogen_clashes = 0
        self.hydrogen_non_clashes = 0
        self.heavy_atom_clashes = 0
        self.heavy_atom_non_clashes = 0

        if not (path_pairs or pathsfile_path):
            raise ValueError("Specify either the path to a file containing the target - ligand path pairs or directly the list of the target -ligand pairs")
        if self.pathsfile_path:
            self.target_ligand_path_pairs = self.load_target_ligand_path_pairs()

        self.construct_complexes()

    def load_target_ligand_path_pairs(self) -> List[Tuple[str,str]]:
        """
        Reads the paths to target, ligand pairs from a file in the form:
            TARGET_PATH_1,LIGAND_PATH_1
            TARGET_PATH_2,LIGAND_PATH_2
            ...

        :return: List[Tuple[str,str]]
        """
        pairs_list = []
        with open(self.pathsfile_path,'r') as file:
            lines = file.readlines()
            for l in lines:
                tuple = l.strip().split(',')
                pairs_list.append(tuple)

        return pairs_list

    def construct_complexes(self):
        """
        based on the paths of target - ligand pairs, construct corresponding Complex objects
        :return:
        """
        logging.info("Constructing the target-ligand complexes, based on the target-ligand path pairs...")
        _targets = {}
        for target_path,ligand_path in tqdm(self.target_ligand_path_pairs):
            if target_path in _targets.keys():
                _target = _targets[target_path]
            else:
                _target = Target(target_path, " ")
                _target.build_kdtrees()
                _targets[target_path] = _target

            _ligand_df = PandasTools.LoadSDF(ligand_path)
            _ligand = Ligand(_ligand_df)
            if _ligand.atom_coords is None:
                logging.warning("Skipped complex.")
                continue

            _complex = Complex(_target,_ligand)
            _complex.check_clashes(treshold=self.clash_treshold)
            self.complexes.append(_complex)

    def evaluate_clashes_over_dataset(self) -> Dict[str,float]:
        """
        Iterate over the complexes in the dataset and read the statements on clashes and produce summary for the dataset
        :return:
        """
        heavy_clashes, heavy_non_clashes, hydrogen_clashes, hydrogen_non_clashes = 0,0,0,0
        for cplx in self.complexes:
            heavy_clashes += cplx.heavy_atom_clash
            heavy_non_clashes += (not cplx.heavy_atom_clash)
            hydrogen_clashes += cplx.hydrogen_clash
            hydrogen_non_clashes += (not cplx.hydrogen_clash)

        data_samples = heavy_clashes+heavy_non_clashes
        heavy_clash_percentage = heavy_clashes/data_samples
        hydrogen_clash_percentage = hydrogen_clashes/data_samples
        assert data_samples == (hydrogen_clashes+hydrogen_non_clashes), "Discrepancy between number of examined heavy atom and hydrogen clashes"
        all_clashes = heavy_clash_percentage + hydrogen_clash_percentage
        retVal = {'data_samples': data_samples, 'heavy_atom_clash_pct': heavy_clash_percentage,
                  'hydrogen_clash_pct': hydrogen_clash_percentage, 'all_available_atom_clash_pct': all_clashes}
        return retVal


    @staticmethod
    def generate_path_file(folder_path: str = None, out_path: str = 'example_files/generated_pathfile.txt'):
        """
        Function to generate the pathfile of target-ligand path pairs, given a directory with prescribed structure,
        the directory is assumed to hold subdirectories, each containing one .pdb file corresponding to the protein and
        several ( num_predicted_poses + 1) .sdf files corresponding to the predicted ligand poses, the top prediction
        is duplicite (rank_1.sdf and rank_1_<confidence_score>.sdf
        :return:
        """
        subdirs = os.listdir(folder_path)
        subdir_paths = [os.path.join(folder_path,p) for p in subdirs]
        output_lines = []

        for target_folder in subdir_paths:
            file_names = os.listdir(target_folder)
            sdfs = 0
            pdbs = 0
            ligand_paths = []

            for f_name in file_names:
                _dot_split = f_name.split('.')
                if _dot_split[-1] == 'sdf':
                    if f_name == "rank1.sdf":
                        logging.debug("Skipping rank1.sdf file as it is assumed to be duplicit.")
                        continue
                    ligand_paths.append(os.path.join(target_folder,f_name))
                    sdfs+=1

                elif _dot_split[-1] == 'pdb':
                    target_path = os.path.join(target_folder,f_name)
                    pdbs+=1

                    if pdbs > 1:
                        raise ValueError(
                            "In the provided folder is more than 1 PDB file, not sure which to use as target.")
                else:
                    logging.warning(f'Non-pdb and non-sdf file {f_name} encountered in the folder {os.path.join(target_folder)}')

            for lig_path in ligand_paths:
                line = target_path+','+lig_path+'\n'
                output_lines.append(line)


        with open(out_path, 'w') as out_file:
            out_file.writelines(output_lines)


if __name__=="__main__":

    args = parser.parse_args()
    logging.basicConfig(level='INFO')

    if args.paths_file is not None:
        dataset = ComplexDataset(pathsfile_path=args.paths_file, clash_treshold=args.treshold)
    else:
        ComplexDataset.generate_path_file(folder_path=args.dataset_folder_path, out_path='example_files/generated_pathfile.txt')
        dataset = ComplexDataset(pathsfile_path='example_files/generated_pathfile.txt', clash_treshold=args.treshold)

    clashes_evaluation = dataset.evaluate_clashes_over_dataset()

    logging.info(f"The examined dataset held {clashes_evaluation['data_samples']} data samples.")
    logging.info(f"The treshold of {args.treshold} Angstrom was used for evaluation of clashes.")
    logging.info(f"Exhibited ligand heavy atom v. target heavy atom steric clashes: {100*clashes_evaluation['heavy_atom_clash_pct']:.2f}%")
    logging.info(f"Exhibited ligand heavy atom v. target hydrogen atom steric clashes: {100*clashes_evaluation['hydrogen_clash_pct']:.2f}%")
    logging.info(f"Exhibited ligand heavy atom v. target all atom steric clashes: {100*clashes_evaluation['all_available_atom_clash_pct']:.2f}%")

    # 1) find out the units in the coordinates (verify for each protein and ligand)
    #   - seems so, checked by peptide bond around 1.35 - consistent (?) with literature: around 1.32
    #TODO:
    # 5) version this project
    # 6) summarize the findings
    # 7) do the checks for some of our results