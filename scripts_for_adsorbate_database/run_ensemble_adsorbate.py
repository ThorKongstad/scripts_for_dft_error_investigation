#partition=katla_verylong
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
from ase.dft.bee import BEEFEnsemble
from gpaw import GPAW, PW, Davidson
from gpaw.utilities import h2gpts
from ase.parallel import parprint, world


def main(db_id:int, db_dir: str = 'molreact.db'):

    # read from  database
    #atoms = read(f'/groups/kemi/thorkong/errors_investigation/molreact.db@id={db_id}')
    if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir))>0 else '.'): raise FileNotFoundError("Can't find database")
    with db.connect(db_dir) as db_obj:
        row = db_obj.get(selection=f'id={db_id}')
        if not row.get('relaxed'): raise ValueError(f"atoms at row id: {db_id} haven't been relaxed.")
        functional = row.get('xc')
        if functional not in ( 'BEEF-vdW', "{'name':'BEEF-vdW','backend':'libvdwxc'}"): raise ValueError(f'row {db_id}, is not a bee functional')
        atoms = row.toatoms()
        structure_str = row.get('structure_str')
        grid_spacing = row.get('grid_spacing')
        if world.rank == 0:
            data_dict = row.get('data')

    parprint(f'outstd of ensemble calculation for db entry {db_id} with structure: {structure_str} and functional: {functional}')

    if not grid_spacing:
        grid_spacing = 0.16
        parprint('grid spacing could not be found in the database entry and was set to 0.16')

    functional_folder = sanitize(functional)
    if world.rank == 0: folder_exist(functional_folder)

    if '{' in functional[0] and '}' in functional[-1] and ':' in functional: functional = eval(functional)

    calc = GPAW(mode=PW(500),
                xc=functional if functional not in ['PBE0'] else {'name': functional, 'backend': 'pw'},
                kpts=[4,4,1],
                basis='dzp',
                txt=f'{functional_folder}/opt_{structure_str}_{db_id}.txt',
                gpts=h2gpts(grid_spacing,atoms.get_cell(),idiv=4),
                parallel={'augment_grids': True, 'sl_auto': True},
                convergence={'eigenstates': 0.000001},
                eigensolver=Davidson(3),
                symmetry='off',
                )

    atoms.set_calculator(calc)
    potential_e = atoms.get_potential_energy()
    ens = BEEFEnsemble(atoms)
    ensem_en_li = ens.get_ensemble_energies()

    if world.rank == 0:
        data_dict.update({'ensemble_en': ensem_en_li})
        update_db(db_dir, dict(id=db_id, ensemble_bool=True, data=data_dict))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_base_id',type=int)
    parser.add_argument('-db','--database',help='directory to the database, if not stated will look for molreact.db in pwd.', default='molreact.db')
    args = parser.parse_args()

    main(args.data_base_id, args.database)