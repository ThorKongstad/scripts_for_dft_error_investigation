#partition=katla_test
#nprocshared=8
#mem=2300MB
#constrain='[v5]'

import argparse
import os
from typing import Tuple, Sequence, NoReturn, Optional, Callable
import numpy as np
from ase.data import reference_states,chemical_symbols
from ase.io.trajectory import Trajectory
from ase.build import bulk#, fcc100, bcc100,hcp0001
import ase.db as db
from ase.parallel import parprint, world
from ase.calculators.mixing import SumCalculator
from ase.calculators.vdwcorrection import vdWTkatchenko09prl
from ase.constraints import ExpCellFilter
from ase.optimize import QuasiNewton
from gpaw import GPAW, PW, Davidson
from gpaw.utilities import h2gpts
from gpaw.analyse.hirshfeld import HirshfeldPartitioning
from gpaw.analyse.vdwradii import vdWradii
from dftd4.ase import DFTD4



def folder_exist(folder_name: str) -> NoReturn:
    if (os.path.basename(folder_name) if '/' in folder_name else folder_name) not in os.listdir(os.path.dirname(folder_name) if '/' in folder_name else './'): os.mkdir(folder_name)

def sanitize(unclean_str: str) -> str:
    for ch in ['!', '*', '?', '{', '[', '(', ')', ']', '}',"'",'.',',']: unclean_str = unclean_str.replace(ch, '')
    for ch in ['/', '\\', '|',' ']: unclean_str = unclean_str.replace(ch, '_')
    for ch in ['=', '+', ':']: unclean_str = unclean_str.replace(ch, '-')
    return unclean_str


def ends_with(string: str, end_str: str) -> str:
    return string + end_str * (end_str != string[-len(end_str):0])


def main(metal: str, functional: str, slab_type: str, guess_lattice: Optional[float] = None, grid_spacing: float = 0.16, vdw_calc: str = 'vdw'):
    at_number = chemical_symbols.index(metal)
    functional_folder = sanitize(functional)
    if world.rank == 0:
        folder_exist(functional_folder)
        folder_exist(f'{functional_folder}/{metal}_latt_fit')

    if guess_lattice is None:
        if slab_type != reference_states[at_number].get('symmetry'): raise ValueError(
            'the given slab type does not match the saved type for ase guess lattice')
        guess_lattice = reference_states[at_number].get('a')

    parprint(f'lattice optimisation for {metal} with {functional}, guess latice is at {guess_lattice}')

    bulk_con = bulk(name=metal, crystalstructure=slab_type, a=guess_lattice)

    calc = GPAW(mode=PW(500),
                xc=functional,
                basis='dzp',
                kpts=(10,10,10),
                txt=f'{functional_folder}/{metal}_latt_fit/lat-opt_{metal}_{slab_type}_a-{lattice}.txt',
                gpts=h2gpts(grid_spacing, bulk_con.get_cell(), idiv=4),
                parallel={'augment_grids': True, 'sl_auto': True},
                convergence={'eigenstates': 0.000001},
                eigensolver=Davidson(3),
                )

    bulk_con.set_calculator(calc)

    if vdw_calc == 'D4':
        bulk_con.calc = SumCalculator([DFTD4(method=functional), calc])
    elif vdw_calc == 'vdw':
        bulk_con.calc = vdWTkatchenko09prl(HirshfeldPartitioning(calc), vdWradii(bulk_con.get_chemical_symbols(), functional))

    CellFilter = ExpCellFilter(bulk_con)
    Optimizer = QuasiNewton(CellFilter)
    traj_obj = Trajectory(f'{ends_with(functional_folder,"/")}{metal}_{slab_type}_{functional}_{vdw_calc}.traj', 'w', bulk_con)
    Optimizer.attach(traj_obj)
    Optimizer.run(fmax=0.001)

    optimal_lattice = np.linalg.norm(bulk_con.cell[0]) * 2 ** 0.5

    if world.rank == 0:
        with db.connect('./lattice_for_functionals.db') as db_obj:
            db_obj.write(bulk_con, metal = metal, type = slab_type, functional=functional,vdw=vdw_calc, lattice=optimal_lattice)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('metal',type=str)
    parser.add_argument('surface_type',type=str,choices=('fcc','bcc','hcp'))
    parser.add_argument('func',type=str)
    parser.add_argument('-vdw','--vdw_calculator',choices=('D4','vdw'),default='vdw')
    parser.add_argument('--lattice','-a',type=float)
    args = parser.parse_args()

    main(metal=args.metal,functional=args.func,slab_type=args.surface_type,guess_lattice=args.lattice,vdw_calc=args.vdw_calculator)