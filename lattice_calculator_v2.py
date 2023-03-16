#partition=katla_test
#nprocshared=8
#mem=2300MB
#constrain='[v5]'

import argparse
import os
from typing import Tuple, Sequence, NoReturn, Optional, Callable
import numpy as np
from ase.data import reference_states,chemical_symbols
from ase.build import bulk#, fcc100, bcc100,hcp0001
import ase.db as db
from ase.parallel import parprint, world
#from gpaw.cluster import Cluster
from gpaw import GPAW, PW, Davidson
from gpaw.utilities import h2gpts
#from collections import namedtuple
from scipy.optimize import curve_fit, minimize, OptimizeResult
import csv
#from kplib import get_kpoints
#from pymatgen.io.ase import AseAtomAdaptor
import pathlib
from time import sleep
from random import randint


def folder_exist(folder_name: str) -> NoReturn:
    if (os.path.basename(folder_name) if '/' in folder_name else folder_name) not in os.listdir(os.path.dirname(folder_name) if '/' in folder_name else './'): os.mkdir(folder_name)

def sanitize(unclean_str: str) -> str:
    for ch in ['!', '*', '?', '{', '[', '(', ')', ']', '}',"'",'.',',']: unclean_str = unclean_str.replace(ch, '')
    for ch in ['/', '\\', '|',' ']: unclean_str = unclean_str.replace(ch, '_')
    for ch in ['=', '+', ':']: unclean_str = unclean_str.replace(ch, '-')
    return unclean_str


#def get_kpts(atoms_obj):
#    structure = AseAtomAdaptor.get_structure(atoms_obj)
#    kpts_dat = get_kpoints(structure, minDistance = 30, include_gamma = False)
#    return kpts_dat['cords']


def secant_method(func: Callable[[float | int], float | int], guess_minus: float | int, guess_current: float | int, maxs_iter: int = 300, con_cri: float | int = 10**(-10)):
    func_eval_minus = func(guess_minus)
    nr_iter = 1

#    BASING THE SECANT METHOD ON THE FORWARD EULER OF THE FUNCTION RESULTS IN TO MUCH FLUCTUATION FOR CONVERGENCE SINCE THE STEPS OF THE SECANT METHOD ARE TO BROAD FOR THE FORWARD EULER TO A GOOD APPROXIMATION
#    forward_euler = lambda x_cur,x_minus,eval_cur,eval_minus: (eval_cur - eval_minus)/(x_cur-x_minus)
#    func_div_eval_minus = forward_euler(guess_minus, guess_minus * 0.95, func_eval_minus, func(guess_minus * 0.95))
#    while abs(func_div_eval_current := forward_euler(guess_current,guess_minus, func_eval_cur := func(guess_current),func_eval_minus)) >= con_cri and nr_iter < maxs_iter:
#        guess_current_temp = guess_current
#        guess_current -= (func_div_eval_current * (guess_current - guess_minus)) / (func_div_eval_current - func_div_eval_minus)
#        guess_minus = guess_current_temp
#        func_eval_minus = func_eval_cur
#        func_div_eval_minus = func_div_eval_current
#        nr_iter += 1
#    return guess_current, nr_iter

    while abs(func_eval_current := func(guess_current)) >= con_cri and nr_iter < maxs_iter:
        guess_current_temp = guess_current
        guess_current -= (func_eval_current * (guess_current - guess_minus)) / (func_eval_current - func_eval_minus)
        guess_minus = guess_current_temp
        func_eval_minus = func_eval_current
        nr_iter += 1
    return guess_current, nr_iter


def calculate_pE_of_latt(lattice: float, metal: str, slab_type:str, functional: str, functional_folder: str, grid_spacing: float) -> float:
    if (isinstance(lattice,list) or isinstance(lattice,tuple)) and len(lattice) == 1: lattice = lattice[0]
    lattice = float(lattice)

    bulk_con = bulk(name=metal, crystalstructure=slab_type, a=lattice)

    calc = GPAW(mode=PW(500),
                xc=functional,
                basis='dzp',
                kpts=(10,10,10),
                txt=f'{functional_folder}/{metal}_latt_fit/lat-opt_{metal}_{slab_type}_a-{lattice}.txt',
                gpts=h2gpts(grid_spacing, bulk_con.get_cell(), idiv=4),
                parallel={'augment_grids': True, 'sl_auto': True},
                convergence={'eigenstates': 0.000001},
                eigensolver=Davidson(3),
                # hund=smile == 'O=O',
                )

    bulk_con.set_calculator(calc)

    potential_energy = bulk_con.get_potential_energy()
    ### DELETE BULK AND CALC ###
    del calc
    del bulk_con
    return potential_energy


def report(res: OptimizeResult) -> NoReturn:
    parprint(f'optimisation: {"succesfull" if res.success else "unsuccesfull"}')
    parprint(f'finale lattice: {res.x}')


def main(metal: str, functional: str, slab_type: str, guess_lattice: Optional[float] = None, grid_spacing: float = 0.16):

    at_number = chemical_symbols.index(metal)
    functional_folder = sanitize(functional)

    script_overlab_protection_time = randint(0, 60)
    if world.rank == 0:
        sleep(script_overlab_protection_time)
        folder_exist(functional_folder)
        folder_exist(f'{functional_folder}/{metal}_latt_fit')
    else:
        sleep(script_overlab_protection_time)

    if guess_lattice is None:
        if slab_type != reference_states[at_number].get('symmetry'): raise ValueError('the given slab type does not match the saved type for ase guess lattice')
        guess_lattice = reference_states[at_number].get('a')

    parprint(f'lattice optimisation for {metal} with {functional}, guess latice is at {guess_lattice}')

    opt_step_func = lambda lat: calculate_pE_of_latt(lat, metal, slab_type, functional, functional_folder, grid_spacing) # this is to make a function which is only dependent on a single variable lat

#    optimised_lat,final_itr = secant_method(opt_step_func,guess_minus= guess_lattice*0.9, guess_current=guess_lattice,maxs_iter=30)
    opt_res = minimize(opt_step_func, x0=guess_lattice, method='Powell', tol=0.01, options=dict(disp=True), bounds=((2.7, 7),))

    report(opt_res)

    if world.rank == 0 and opt_res.success:
        if 'lattice_calc.csv' not in os.listdir(): pathlib.Path('lattice_calc.csv').touch()
        with open('lattice_calc.csv','a') as csv_file:
            fields = ['metal', 'type', 'functional','lattice']
            writer_obj = csv.DictWriter(csv_file,fieldnames=fields)
            writer_obj.writerow(
                dict(
                    metal=metal,
                    type=slab_type,
                    functional=functional,
                    lattice=opt_res.x if not isinstance(opt_res.x, list) else opt_res.x[0]
                )
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('metal',type=str)
    parser.add_argument('surface_type',type=str,choices=('fcc','bcc','hcp'))
    parser.add_argument('func',type=str)
    parser.add_argument('--lattice','-a',type=float)
    args = parser.parse_args()

    main(metal=args.metal,functional=args.func,slab_type=args.surface_type,guess_lattice=args.lattice)