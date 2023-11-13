import argparse
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from scripts_for_molecule_database_calculations import sanitize, folder_exist

# from ase.io import *
import ase.db as db
from ase.data.pubchem import pubchem_atoms_search
from ase.build import molecule
from gpaw.cluster import Cluster
# from gpaw import GPAW, PW, Davidson
# from gpaw.utilities import h2gpts
# import numpy as np


def main(smile: str, functional: str, db_dir: str = 'molreact.db', setup_path: str | bool = False, grid_spacing: float = 0.16):
    # create ase mol
    if smile == '[HH]': atoms = Cluster(molecule('H2'))  # pubchem search hates hydrogen, it hates its name, it hates its cid and most of all it hates its weird smile and dont you dare confuse HH for hydrogen
    elif 'cid' in smile: atoms = Cluster(pubchem_atoms_search(cid=int(smile.replace('cid', ''))))
    else: atoms = Cluster(pubchem_atoms_search(smiles=smile))
    atoms.minimal_box(border=6, h=grid_spacing, multiple=4)

    # connect to db

    with db.connect(db_dir) as db_obj:
        # db_id = db_obj.reserve(xc = functional, smiles=smile)
        db_obj.write(atoms=atoms, xc=functional, smiles=smile, relaxed=False, vibration=False, grid_spacing=grid_spacing, setup=setup_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('smiles_str')
    parser.add_argument('functional', help='str denoting what fucntional to calculate with')
    parser.add_argument('-db', '--database', help='name or directory for the database, if not stated will make a molreact.db in pwd.', default='molreact.db')
    parser.add_argument('--setup', '-s', default=False)
    parser.add_argument('--grid_spacing', '-g', default=0.16)
    args = parser.parse_args()

    main(smile=args.smiles_str, functional=args.functional, setup_path=args.setup, grid_spacing=args.grid_spacing, db_dir=args.database)
