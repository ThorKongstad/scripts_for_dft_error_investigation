#partition=katla
#nprocshared=32
#mem=2300MB
#constrain='[v1|v2|v3|v4|v5]'

import argparse
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from scripts_for_adsorbate_database import sanitize, folder_exist, update_db

import ase.db as db
from gpaw import GPAW, PW, Davidson
from gpaw.utilities import h2gpts
from gpaw.utilities.timing import ParallelTimer
from ase.parallel import parprint, world, barrier


def main(db_id:int, db_dir: str = 'molreact.db'):
    # read from  database
    #atoms = read(f'/groups/kemi/thorkong/errors_investigation/molreact.db@id={db_id}')
    if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir))>0 else '.'): raise FileNotFoundError("Can't find database")
    with db.connect(db_dir) as db_obj:
        row = db_obj.get(selection=f'id={db_id}')
        functional = row.get('xc')
        atoms = row.toatoms()
        structure_str = row.get('structure_str')
        grid_spacing= row.get('grid_spacing')

    parprint(f'outstd of single point timing calculation for db entry {db_id} with structure: {structure_str} and functional: {functional}')

    if not grid_spacing:
        grid_spacing = 0.16
        parprint('grid spacing could not be found in the database entry and was set to 0.16')

    functional_folder = sanitize(functional)
    if world.rank == 0:
        folder_exist(functional_folder)
        folder_exist(functional_folder + f'/sp_timings_{structure_str}_{db_id}')

    if '{' in functional[0] and '}' in functional[-1] and ':' in functional: functional = eval(functional)

    barrier()

    calc = GPAW(mode=PW(500),
                xc=functional if functional not in ['PBE0'] else {'name': functional, 'backend': 'pw'},
                kpts=[4,4,1],
                basis='dzp',
                txt=f'{functional_folder}/sp_timings_{structure_str}_{db_id}/sp_{structure_str}_{db_id}.txt',
                gpts=h2gpts(grid_spacing, atoms.get_cell(), idiv=4),
                parallel={'augment_grids': True, 'sl_auto': True},
                convergence={'eigenstates': 0.000001},
                eigensolver=Davidson(3),
                symmetry='off',
                timer=ParallelTimer(prefix=f'{functional_folder}/sp_timings_{structure_str}_{db_id}/timings'),
                )

    atoms.set_calculator(calc)

    atoms.get_potential_energy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_base_id',type=int)
    parser.add_argument('-db','--database',help='directory to the database, if not stated will look for molreact.db in pwd.', default='molreact.db')
    args = parser.parse_args()

    main(args.data_base_id, args.database)
