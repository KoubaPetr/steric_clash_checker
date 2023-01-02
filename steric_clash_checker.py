import Bio.PDB
import numpy as np
import logging
import os
import argparse
from rdkit.Chem import PandasTools
from rdkit.Chem.rdchem import Conformer
from pandas import DataFrame as df
from sklearn.neighbors import KDTree as KDTree
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any
from tqdm import tqdm

"""

TODO: Describe this file 

"""

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_folder_path", default='data/user_predictions_testset', type=str, help="Path to the folder containing the subfolders holding the target PDB file and corresponding ligand pose .sdf files.")
parser.add_argument("--paths_file", default=None, type=str, help="Path to the file containing the target-ligand path pairs.")
parser.add_argument("--treshold", default=0.4, type=float, help="Max distance between target-ligand atoms in Angstrom, so they are considered as clashing.")
parser.add_argument('--exclude_files', nargs='+', default=['rank1.sdf'])
parser.add_argument('--log_to', type=str, default=None, help="Name of the file into which the logs should be printed. If not provided, it is printed into console.")
parser.add_argument('--max_ranking', type=int, default=None, help="Name of the file into which the logs should be printed. If not provided, it is printed into console.")

@dataclass
class Ligand:
    parsed_df: df
    name: str = None
    num_heavy_atoms: int = None
    atom_coords: np.ndarray = None
    atom_ids: np.ndarray = None
    ranking: int = None

    def __post_init__(self) -> None:
        try:
            self.num_heavy_atoms = self.parsed_df['ROMol'][0].GetNumHeavyAtoms()
            if not len(self.parsed_df['ROMol'][0].GetAtoms())==self.parsed_df['ROMol'][0].GetNumHeavyAtoms():
                raise ValueError("There are some unhandled hydrogens in the ligand!")
            conformer: Conformer = self.parsed_df['ROMol'][0].GetConformer() #TODO: verify if the hydrogens are not messing up the order
            self.atom_coords = np.array([conformer.GetAtomPosition(a_idx) for a_idx in range(self.num_heavy_atoms)])
            self.atom_ids = np.array([a.GetIdx() for a in self.parsed_df['ROMol'][0].GetAtoms()])
            self.ranking = int(self.name[4:]) #This assumes that the name of the ligand is in the form 'rank{n}'
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
        self.heavy_atom_id = []
        self.hydrogen_id = []
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
                    self.heavy_atom_id.append(atom.full_id)
                else:
                    self.hydrogen_coords.append(atom_xyz)
                    self.hydrogen_id.append(atom.full_id)
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
    heavy_mindist: float = None
    hydrogen_mindist: float = None
    hydrogen_nn_dists: np.ndarray = None
    hydrogen_nn_idcs: np.ndarray = None
    heavy_nn_dists: np.ndarray = None
    heavy_nn_idcs: np.ndarray = None
    heavy_atom_clash: bool = None
    hydrogen_clash: bool = None
    last_clash_treshold: float = None
    clash_data: Dict[str,List[Tuple[Tuple[Any],int]]] = field(default_factory=dict)

    def __post_init__(self):
        """
        To compute the nearest neighbors upon instantiation
        :return:
        """
        self.hydrogen_nn_dists, self.hydrogen_nn_idcs = self.target.hydrogen_kdtree.query(self.ligand.atom_coords)
        self.heavy_nn_dists, self.heavy_nn_idcs = self.target.heavy_atom_kdtree.query(self.ligand.atom_coords)
        self.heavy_mindist = self.heavy_nn_dists.min()
        self.hydrogen_mindist = self.hydrogen_nn_dists.min()
        self.clash_data['Heavy_atom'] = []
        self.clash_data['Hydrogen'] = []

    def check_clashes(self, treshold: float = 0.4) -> Tuple[bool,bool]:
        """
        Function to check steric clashes given a treshold on a minimum allowed distance
        of a ligand atom (only heavy atoms for now) and target atoms (both heavy and hydrogen atoms).

        :param treshold: in Angstroms, minimum allowed distance for non-clashing atoms,
                         default value = 0.4A (same as DiffDock paper)
        :return:  Tuple[bool, bool] - bool values for hydrogen/heavy atoms clashes
        """
        self.hydrogen_clash = self.hydrogen_mindist < treshold
        self.heavy_atom_clash = self.heavy_mindist < treshold
        self.last_clash_treshold = treshold
        if self.heavy_atom_clash:
            for heavy_atom_idx, ligand_atom_idx in zip(self.heavy_nn_idcs[(self.heavy_nn_dists<treshold)[:,0]], self.ligand.atom_ids[(self.heavy_nn_dists<treshold)[:,0]]):
                _clash_data = self.target.heavy_atom_id[heavy_atom_idx[0]], self.ligand.atom_ids[ligand_atom_idx], self.heavy_mindist
                self.clash_data['Heavy_atom'].append(_clash_data)
        if self.hydrogen_clash:
            for hydrogen_idx, ligand_atom_idx in zip(self.hydrogen_nn_idcs[self.hydrogen_nn_dists < treshold],
                                                       self.ligand.atom_ids[(self.hydrogen_nn_dists < treshold)[:,0]]):
                _clash_data = self.target.hydrogen_id[hydrogen_idx], self.ligand.atom_ids[ligand_atom_idx], self.hydrogen_mindist
                self.clash_data['Hydrogen'].append(_clash_data)
        return self.hydrogen_clash, self.heavy_atom_clash

    def report_clashes_for_complex(self) -> List[str]:
        retVal = []

        for _clash_data in self.clash_data['Heavy_atom']:
            _msg = f"Heavy atom clash: Target {self.target.name}, atom {_clash_data[0]} and Ligand {self.ligand.name}, atom {_clash_data[1]}, distance = {_clash_data[2]:.3f}"
            retVal.append(_msg)

        for _clash_data in self.clash_data['Hydrogen']:
            _msg = f"Hydrogen clash: Target {self.target.name}, atom {_clash_data[0]} and Ligand {self.ligand.name}, atom {_clash_data[1]}, distance = {_clash_data[2]:.3f}"
            retVal.append(_msg)
        return retVal


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
                _target_name = target_path.split('/')[-1].split('.')[-2].split('_')[0] #Also this assumes particular naming of the pdb file */protein_name_*.pdb
                _target = Target(target_path, _target_name)
                _target.build_kdtrees()
                _targets[target_path] = _target

            _ligand_df = PandasTools.LoadSDF(ligand_path)
            _ligand_name = ligand_path.split('/')[-1].split('_')[0] #This requires standartization of the file names - for now we name the ligand just by its rank according to the confidence - e.g. we assume to pass here name='rank2'
            _ligand = Ligand(_ligand_df, name=_ligand_name)
            if _ligand.atom_coords is None:
                logging.warning("Skipped complex.")
                continue

            _complex = Complex(_target,_ligand)
            _complex.check_clashes(treshold=self.clash_treshold)
            self.complexes.append(_complex)

    def evaluate_clashes_over_dataset(self, max_ranking: int = None) -> Dict[str,float]:
        """
        Iterate over the complexes in the dataset and read the statements on clashes and produce summary for the dataset
        :return:
        """
        heavy_clashes, heavy_non_clashes, hydrogen_clashes, hydrogen_non_clashes = 0,0,0,0
        for cplx in self.complexes:
            if max_ranking is not None and cplx.ligand.ranking < max_ranking:
                heavy_clashes += cplx.heavy_atom_clash
                heavy_non_clashes += (not cplx.heavy_atom_clash)
                hydrogen_clashes += cplx.hydrogen_clash
                hydrogen_non_clashes += (not cplx.hydrogen_clash)

        data_samples = heavy_clashes+heavy_non_clashes
        heavy_clash_percentage = heavy_clashes/data_samples if data_samples > 0 else 0
        hydrogen_clash_percentage = hydrogen_clashes/data_samples if data_samples > 0 else 0
        assert data_samples == (hydrogen_clashes+hydrogen_non_clashes), "Discrepancy between number of examined heavy atom and hydrogen clashes"
        all_clashes = heavy_clash_percentage + hydrogen_clash_percentage
        retVal = {'data_samples': data_samples, 'heavy_atom_clash_pct': heavy_clash_percentage,
                  'hydrogen_clash_pct': hydrogen_clash_percentage, 'all_available_atom_clash_pct': all_clashes}
        return retVal

    def report_clashes(self):
        all_clashes = ["-----------------------REPORTING CLASHES-----------------------\n"]
        for cplx in self.complexes:
            all_clashes.extend(cplx.report_clashes_for_complex())

        for msg in all_clashes:
            logging.info(msg)

    @staticmethod
    def generate_path_file(folder_path: str = None, exclude_files: List[str] = None, out_path: str = 'data/generated_pathfile.txt'):
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
                    if f_name in exclude_files:
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
    if args.log_to:
        logging.basicConfig(filename=f'logs/{args.log_to}', filemode='w', encoding='utf-8', level='INFO')
    else:
        logging.basicConfig(level='INFO')

    if args.paths_file is not None:
        dataset = ComplexDataset(pathsfile_path=args.paths_file, clash_treshold=args.treshold)
    else:
        ComplexDataset.generate_path_file(folder_path=args.dataset_folder_path, exclude_files=args.exclude_files, out_path='data/generated_pathfile.txt')
        dataset = ComplexDataset(pathsfile_path='data/generated_pathfile.txt', clash_treshold=args.treshold)

    clashes_evaluation = dataset.evaluate_clashes_over_dataset(max_ranking=args.max_ranking)

    logging.info(f"The examined dataset held {clashes_evaluation['data_samples']} data samples.")
    logging.info(f"The treshold of {args.treshold} Angstrom was used for evaluation of clashes.")
    _max_ranking_msg = f' (considering top {args.max_ranking} poses)' if args.max_ranking else ""
    logging.info(f"Exhibited ligand heavy atom v. target heavy atom steric clashes: {100*clashes_evaluation['heavy_atom_clash_pct']:.2f}%{_max_ranking_msg}")
    logging.info(f"Exhibited ligand heavy atom v. target hydrogen atom steric clashes: {100*clashes_evaluation['hydrogen_clash_pct']:.2f}%")
    logging.info(f"Exhibited ligand heavy atom v. target all atom steric clashes: {100*clashes_evaluation['all_available_atom_clash_pct']:.2f}%")

    dataset.report_clashes()

    #TODO: log the min distances for each target ligand pair - DONE
    #TODO: save the target and ligand names - DONE
    #TODO: log by percentiles (top10 poses etc) - DONE
    #TODO: check their method
    #TODO: run with our data
    #TODO: log the examples of clashing atom pairs - DONE
    #TODO: summarize into the google slides
