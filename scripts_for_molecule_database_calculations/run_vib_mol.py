#partition=katla_medium
#nprocshared=8
#mem=2300MB
#constrain='[v4|v5]'

import argparse
import os
import ase.db as db
from ase.vibrations import Vibrations
from gpaw import GPAW, PW, Davidson
from gpaw.utilities import h2gpts
from typing import NoReturn
from ase.parallel import parprint, world


def folder_exist(folder_name: str, path:str = '.') -> NoReturn:
    if folder_name not in os.listdir(path): os.mkdir(ends_with(path,'/')+folder_name)

def file_nexist(file_name: str, path:str = '.', rm_flags='',return_path:bool=False) -> NoReturn|str:
    if file_name not in os.listdir(path):
        os.system(f'rm {rm_flags} "{ends_with(path,"/")}{file_name}"')
        os.wait()
    if return_path: return f'{ends_with(path,"/")}{file_name}'

def ends_with(st:str, end:str):
    if st[-1] != end: return st+end
    else: return st

def sanitize(unclean_str: str) -> str:
    for ch in ['!', '*', '?', '{', '[', '(', ')', ']', '}',"'",'"']: unclean_str = unclean_str.replace(ch, '')
    for ch in ['/', '\\', '|',' ']: unclean_str = unclean_str.replace(ch, '_')
    for ch in ['=', '+', ':']: unclean_str = unclean_str.replace(ch, '-')
    return unclean_str

def clean_old_files(db_id,smile,functional_folder):
    os.getcwd()
    if (old_fil := f'vib_{smile}_{db_id}.txt') in os.listdir(functional_folder):
        folder_exist(f'old_vibs',path=functional_folder)
        os.system(f'mv -f "{ends_with(functional_folder,"/")}{old_fil}" "{file_nexist(old_fil,f"{functional_folder}/old_vibs",return_path=True)}"')
        os.wait()
    if (old_folder := f'vib_{smile}_{db_id}') in os.listdir(functional_folder):
        folder_exist(f'old_vibs',path=functional_folder)
        os.system(f'mv -f "{ends_with(functional_folder,"/")}{old_folder}" {file_nexist(old_folder,f"{functional_folder}/old_vibs",rm_flags="-r",return_path=True)}')
        os.wait()

def main(db_id:int, clean_old:bool=True, db_dir: str = 'molreact.db'):
    # read from  database
    if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir))>0 else '.'): raise FileNotFoundError("Can't find database")
    with db.connect(db_dir) as db_obj:
        row = db_obj.get(selection=f'id={db_id}')
        if row.get('relaxed'): atoms = row.toatoms()
        else: raise f"atoms at row id: {db_id} haven't been relaxed."
        smile = row.get('smiles')
        functional = row.get('xc')
        grid_spacing= row.get('grid_spacing')

    parprint(f'outstd of vib calculation for db entry {db_id} with structure: {smile} and functional: {functional}')

    if not grid_spacing:
        grid_spacing = 0.16
        parprint('grid spacing could not be found in the database entry and was set to 0.16')

    functional_folder = sanitize(functional)
    folder_exist(functional_folder)


    if world.rank == 0 and clean_old: clean_old_files(db_id,smile,functional_folder)

    if '{' in functional[0] and '}' in functional[-1] and ':' in functional: functional = eval(functional)

    calc = GPAW(mode=PW(500),
                xc=functional,
                basis='dzp',
                txt=f'{functional_folder}/vib_{smile}_{db_id}.txt',
                gpts=h2gpts(grid_spacing, atoms.get_cell(), idiv=4),
                parallel={'augment_grids': True, 'sl_auto': True},
                convergence={'eigenstates': 0.000001},
                eigensolver=Davidson(3),
                symmetry='off',
                hund=smile == 'O=O',
                )

    atoms.set_calculator(calc)
    # run vib calc
    vib = Vibrations(atoms,name=f'{functional_folder}/vib_{smile}_{db_id}')
    vib.run()
    #vib_dat = vib.get_vibrations()

    if world.rank == 0:
        vib.summary(log=f'{functional_folder}/vib_en_{smile}_{db_id}.txt')

        with open(f'{functional_folder}/vib_en_{smile}_{db_id}.txt','r') as fil:
            energy_string = fil.read()

        # saving vib data
        with db.connect(db_dir) as db_obj:
            db_obj.update(db_id, vibration=True, vib_en=energy_string)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_base_id',type=int)
    parser.add_argument('-db','--database',help='directory to the database, if not stated will look for molreact.db in pwd.', default='molreact.db')
    args = parser.parse_args()

    main(args.data_base_id, db_dir=args.database)