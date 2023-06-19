import argparse
import os
import sys
from typing import NoReturn, Tuple
sys.path + os.path.basename(__file__)
#from scripts_for_adsorbate_database import sanitize, folder_exist

from ase.io import read
from ase import Atoms
import ase.db as db


def main(traj_structure:str, structure_str: str, functional_str: str,  db_dir: str, grid_spacing: float = 0.16):
    atoms: Atoms = read(traj_structure)

    with db.connect(db_dir) as db_obj:
        db_obj.write(atoms=atoms, xc=functional_str, structure_str=structure_str, relaxed=False, vibration=False, grid_spacing=grid_spacing)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('struc_traj')
    parser.add_argument('struc_str')
    parser.add_argument('functional',help='str denoting what functional to calculate with')
    parser.add_argument('db',help='name or directory for the database.')
    parser.add_argument('--grid_spacing', '-g', default=0.16)
    args = parser.parse_args()

    main(traj_structure=args.struc_traj, structure_str=args.struc_str, functional_str=args.functional, db_dir=args.db, grid_spacing=args.grid_spacing)
