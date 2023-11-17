import argparse
#sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
#from scripts_for_adsorbate_database import sanitize, folder_exist

from ase.io import read, write
from ase.constraints import FixedLine
from ase import Atoms


def main(traj_structure:str, locked_index: tuple[int,int], structure_str: str, functional_str: str,  db_dir: str, grid_spacing: float = 0.16):
    atoms: Atoms = read(traj_structure)
    atoms.set_constraint(constraints=[Atoms.constraints] + [FixedLine(indices=locked_index, direction=(0, 0, 1))])
    write(traj_structure,atoms)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('struc_traj')
    parser.add_argument('locked_index', nargs=2, type=int)
    args = parser.parse_args()

    main(traj_structure=args.struc_traj, locked_index=args.locked_index)
