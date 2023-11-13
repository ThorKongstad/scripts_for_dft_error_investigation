#partition=katla_medium
#nprocshared=8
#mem=2300MB
#constrain='[v4|v5]'

import argparse
import os
import time
import sys
import pathlib
from subprocess import call
from typing import NoReturn

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from scripts_for_molecule_database_calculations import update_db, folder_exist, sanitize, ends_with

import ase.db as db
from ase.vibrations import Vibrations
from gpaw import GPAW, PW, Davidson
from gpaw.utilities import h2gpts
from ase.parallel import parprint, world, barrier


def file_dont_exist(file_name: str, path: str = '.', rm_flags='', return_path: bool = False) -> NoReturn | str:
    if file_name in os.listdir(path):
        call(['rm', f'{rm_flags}', f"'{ends_with(path, '/')}{file_name}'"])
    if return_path: return f'{ends_with(path, "/")}{file_name}'


def clean_old_files(functional_folder, file_name):
    os.getcwd()
    if file_name in os.listdir(functional_folder):
        folder_exist(f'old_vibs', path=functional_folder)
        call(['mv', '-f', f"{ends_with(functional_folder, '/')}{file_name}", f"{file_dont_exist(file_name, f'{functional_folder}/old_vibs', return_path=True)}"])
    if (old_folder := file_name.replace('.txt', '')) in os.listdir(functional_folder):
        folder_exist(f'old_vibs', path=functional_folder)
        call(['mv', '-f', f"{ends_with(functional_folder, '/')}{old_folder}", f"{file_dont_exist(old_folder, f'{functional_folder}/old_vibs', rm_flags='-r', return_path=True)}"])


def main(db_id: int, clean_old: bool = True, db_dir: str = 'molreact.db'):
    # read from  database
    if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir)) > 0 else '.'): raise FileNotFoundError("Can't find database")
    with db.connect(db_dir) as db_obj:
        row = db_obj.get(selection=f'id={db_id}')
        if row.get('relaxed'): atoms = row.toatoms()
        else: raise f"atoms at row id: {db_id} haven't been relaxed."
        smile = row.get('smiles')
        functional = row.get('xc')
        grid_spacing = row.get('grid_spacing')

    parprint(f'outstd of vib calculation for db entry {db_id} with structure: {smile} and functional: {functional}')

    if not grid_spacing:
        grid_spacing = 0.16
        parprint('grid spacing could not be found in the database entry and was set to 0.16')

    functional_folder = sanitize(functional)
    if world.rank == 0: folder_exist(functional_folder)
    file_name = f'vib_{smile}_{db_id}.txt'

    if world.rank == 0 and clean_old: clean_old_files(functional_folder, file_name)

    if '{' in functional[0] and '}' in functional[-1] and ':' in functional: functional = eval(functional)

    barrier()

    calc = GPAW(mode=PW(500),
                xc=functional,
                basis='dzp',
                txt=f'{functional_folder}/{file_name}',
                gpts=h2gpts(grid_spacing, atoms.get_cell(), idiv=4),
                parallel={'augment_grids': True, 'sl_auto': True},
                convergence={'eigenstates': 0.000001},
                eigensolver=Davidson(3),
                symmetry='off',
                hund=smile == 'O=O',
                )

    atoms.set_calculator(calc)
    # run vib calc
    vib = Vibrations(atoms, name=f'{functional_folder}/{file_name.replace(".txt", "")}')
    vib.run()
    #vib_dat = vib.get_vibrations()

    if world.rank == 0:
        vib.summary(log=f'{functional_folder}/{file_name.replace("vib", "vib_en")}')

        with open(f'{functional_folder}/{file_name.replace("vib", "vib_en")}', 'r') as fil:
            energy_string = fil.read()

        # saving vib data
        update_db(db_dir, db_update_args=dict(
            id=db_id,
            vibration=True,
            vib_en=energy_string
        ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_base_id', type=int)
    parser.add_argument('-db', '--database', help='directory to the database, if not stated will look for molreact.db in pwd.', default='molreact.db')
    args = parser.parse_args()

    main(args.data_base_id, db_dir=args.database)
