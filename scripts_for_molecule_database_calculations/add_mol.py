import argparse
import os
# from ase.io import *
import ase.db as db
from ase.data.pubchem import pubchem_atoms_search
from ase.build import molecule
from gpaw.cluster import Cluster
# from gpaw import GPAW, PW, Davidson
# from gpaw.utilities import h2gpts
# import numpy as np
from typing import NoReturn


def folder_exist(folder_name: str) -> NoReturn:
    if folder_name not in os.listdir(): os.mkdir(folder_name)


def sanitize(unclean_str: str) -> str:
    for ch in ['!', '*', '?', '{', '[', '(', ')', ']', '}', "'",'"']: unclean_str = unclean_str.replace(ch, '')
    for ch in ['/', '\\', '|',' ']: unclean_str = unclean_str.replace(ch, '_')
    for ch in ['=', '+', ':']: unclean_str = unclean_str.replace(ch, '-')
    return unclean_str


def main(smile: str, functional: str, db_dir: str = 'molreact.db', setup_path: str | None = None, grid_spacing: float = 0.16) -> NoReturn:
    # create ase mol
    if smile == '[HH]': atoms = Cluster(molecule('H2'))  # pubchem search hates hydrogen, it hates its name, it hates its cid and most of all it hates its weird smile and dont you dare confuse HH for hydrogen
    elif 'cid' in smile: atoms = Cluster(pubchem_atoms_search(cid=int(smile.replace('cid',''))))
    else: atoms = Cluster(pubchem_atoms_search(smiles=smile))
    atoms.minimal_box(border=6, h=grid_spacing, multiple=4)

    # connect to db

    with db.connect(db_dir) as db_obj:
        # db_id = db_obj.reserve(xc = functional, smiles=smile)
        db_obj.write(atoms=atoms, xc=functional, smiles=smile, relaxed=False, vibration=False, grid_spacing=grid_spacing, setup=setup_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('smiles_str')
    parser.add_argument('functional',help='str denoting what fucntional to calculate with')
    parser.add_argument('-db','--database',help='name or directory for the database, if not stated will make a molreact.db in pwd.', default='molreact.db')
    parser.add_argument('--setup', '-s')
    parser.add_argument('--grid_spacing', '-g')
    args = parser.parse_args()

    main(smile=args.smiles_str, functional=args.functional, setup_path=args.setup, grid_spacing=args.grid_spacing, db_dir=args.database)
