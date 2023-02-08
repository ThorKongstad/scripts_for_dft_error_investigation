#partition=katla_test
#nprocshared=8
#mem=2300MB
#constrain='[v5]'

import argparse
import os
from typing import Tuple, Sequence, NoReturn
import math
from ase.data import reference_states,chemical_symbols
from ase.build import bulk#, fcc100, bcc100,hcp0001
import ase.db as db
from ase.parallel import parprint, world
#from gpaw.cluster import Cluster
from gpaw import GPAW, PW, Davidson
from gpaw.utilities import h2gpts
#from collections import namedtuple
from scipy.optimize import curve_fit
import numpy as np

def folder_exist(folder_name: str) -> NoReturn:
    if (os.path.basename(folder_name) if '/' in folder_name else folder_name) not in os.listdir(os.path.dirname(folder_name) if '/' in folder_name else './'): os.mkdir(folder_name)

def sanitize(unclean_str: str) -> str:
    for ch in ['!', '*', '?', '{', '[', '(', ')', ']', '}',"'",'.',',']: unclean_str = unclean_str.replace(ch, '')
    for ch in ['/', '\\', '|',' ']: unclean_str = unclean_str.replace(ch, '_')
    for ch in ['=', '+', ':']: unclean_str = unclean_str.replace(ch, '-')
    return unclean_str

def main(metal:str,functional:str,slab_type:str,guess_lattice:float|None=None, grid_spacing:float=0.16):

    at_number = chemical_symbols.index(metal)
    functional_folder = sanitize(functional)
    folder_exist(functional_folder)
    folder_exist(f'{functional_folder}/{metal}_latt_fit')

    if guess_lattice is None:
        if slab_type != reference_states[at_number].get('symmetry'): raise ValueError('the given slab type does not match the saved type for ase guess lattice')
        guess_lattice = reference_states[at_number].get('a')

    parprint(f'lattice optimisation for {metal} with {functional}, guess lattice is at {guess_lattice}')

    spinpol_dic = {'spinpol':True,'hund':True} if metal in ['Ni','Co','Fe'] else {}

    potential_energies = []
    lattices = [guess_lattice * 0.75,guess_lattice * 0.85, guess_lattice * 0.95, guess_lattice , guess_lattice * 1.05, guess_lattice * 1.15,guess_lattice * 1.25]
    for cur_lattice in lattices:

        #surface_builder = fcc100 if slab_type == 'fcc' else bcc100 if slab_type == 'bcc' else hcp0001 if slab_type == 'hcp' else None
        #bulk_con = surface_builder(symbol=metal, a=cur_lattice, size=(1, 1, 4),periodic=True)
        #bulk_con = Cluster(bulk_con)
        #bulk_con.minimal_box(border=[cur_lattice,cur_lattice,cur_lattice*4],h=grid_spacing,multiple=4)

        bulk_con = bulk(name=metal,crystalstructure=slab_type,a=cur_lattice)

        calc = GPAW(mode=PW(500),
                    xc=functional,
                    basis='dzp',
                    kpts = (10,10,10),
                    txt=f'{functional_folder}/{metal}_latt_fit/lat-opt_{metal}_{slab_type}_a-{cur_lattice}.txt',
                    gpts=h2gpts(grid_spacing, bulk_con.get_cell(), idiv=4),
                    parallel={'augment_grids': True, 'sl_auto': True},
                    convergence={'eigenstates': 0.000001},
                    eigensolver=Davidson(3),
                    **spinpol_dic)

        bulk_con.set_calculator(calc)

        potential_energy = bulk_con.get_potential_energy()
        potential_energies.append(potential_energy)
        ### DELETE BULK AND CALC ###
        del calc
        del bulk_con

#    parable = lambda x,a,b,c: a*x**2+b*x+c
    morse_potential = lambda x,D,a,r: D*(1-np.exp(-a*(x-r)))**2

    param, param_cov = curve_fit(morse_potential,lattices,potential_energies)

    optimal_lattice = (-param[1])/(2*param[0]) # parable top point x = -b/2a
    estimated_energy = morse_potential(optimal_lattice,*param)

    #surface_builder = fcc100 if slab_type == 'fcc' else bcc100 if slab_type == 'bcc' else hcp0001 if slab_type == 'hcp' else None
    #bulk_con = surface_builder(symbol=metal, a=optimal_lattice, size=(1, 1, 4))

    bulk_con = bulk(name=metal, crystalstructure=slab_type, a=optimal_lattice)

    calc = GPAW(mode=PW(500),
                xc=functional,
                basis='dzp',
                kpts=(10,10,10),
                txt=f'{functional_folder}/lat-opt_{metal}_{slab_type}_a-{optimal_lattice}.txt',
                gpts=h2gpts(grid_spacing, bulk_con.get_cell(), idiv=4),
                parallel={'augment_grids': True, 'sl_auto': True},
                convergence={'eigenstates': 0.000001},
                eigensolver=Davidson(3),
                **spinpol_dic)

    bulk_con.set_calculator(calc)
    opt_potential = bulk_con.get_potential_energy()
    # test that the new potential energy is better than the others

    if any((opt_potential > p for p in potential_energies)):
        raise ValueError('lattice estimation went wrong and the resulting lattice constant is worse than one of the ones tried.')

    parprint(f'parabola param:')
    parprint(param)
    parprint('')
    parprint('parabola covariance:')
    parprint(param_cov)
    parprint('')
    parprint('lattices:')
    parprint(lattices)
    parprint(f'estimated lattice: {optimal_lattice}')
    parprint('')
    parprint('potential energies')
    parprint(potential_energies)
    parprint(f'estimated: {estimated_energy}, calculated: {bulk_con.get_potential_energy()}')

    # setup surface
    # get lattice constant
    # calculate surface
    # save potential energy
    # make sure old atoms module plus calculater is deleted
    # setup first modified surface with +5% lattice constant
    # calculate surface
    # save potential energy
    # make sure old atoms module plus calculater is deleted
    # setup first modified surface with -5% lattice constant
    # calculate surface
    # save potential energy
    # make sure old atoms module plus calculater is deleted
    # fit a parable to the 3 potential energies
    # get top of parable
    # use that for fourth surface
    # calculate potential energy
    # report deviance between expected energy from parable and calculate energy
    # save to a database

    if world.rank == 0:
        with db.connect('/groups/kemi/thorkong/errors_investigation/slab_calc/lattice_for_functionals.db') as db_obj:
            db_obj.write(bulk_con, functional=functional, lattice=optimal_lattice, data={'lattice_test':{'a':lattices, 'energies':potential_energies, 'parabola': {
                'D_e':param[0],'a':param[1],'r_e':param[2]}, 'parabola_covariance':param_cov}})



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('metal',type=str)
    parser.add_argument('surface_type',type=str,choices=('fcc','bcc','hcp'))
    parser.add_argument('func',type=str)
    parser.add_argument('--lattice','-a',type=float)
    args = parser.parse_args()

    main(metal=args.metal,functional=args.func,slab_type=args.surface_type,guess_lattice=args.lattice)