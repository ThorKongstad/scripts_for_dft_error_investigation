#partition=katla_day
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
#from kplib import get_kpoints
#from pymatgen.io.ase import AseAtomAdaptor
import time


def folder_exist(folder_name: str, path: str = '.', tries: int = 10) -> NoReturn:
    try:
        tries -= 1
        if folder_name not in os.listdir(path): os.mkdir(ends_with(path, '/')+folder_name)
    except FileExistsError:
        time.sleep(2)
        if tries > 0: folder_exist(folder_name, path=path, tries=tries)

def ends_with(string: str, end_str: str) -> str:
    return string + end_str * (end_str != string[-len(end_str):0])

def sanitize(unclean_str: str) -> str:
    for ch in ['!', '*', '?', '{', '[', '(', ')', ']', '}',"'",'"','.']: unclean_str = unclean_str.replace(ch, '')
    for ch in ['/', '\\', '|',' ']: unclean_str = unclean_str.replace(ch, '_')
    for ch in ['=', '+', ':',',']: unclean_str = unclean_str.replace(ch, '-')
    return unclean_str

def get_parable(points: Sequence[Tuple[float,float]]) -> Tuple[float,float,float]:
    # an error in here
    a = (points[2][0]*(points[1][1]-points[0][1]) + points[1][0]*(points[1][1]-points[2][1]) + points[0][0]*(points[0][1]-points[1][1]))/((points[0][0]-points[1][0])*(points[0][0]-points[2][0])*(points[1][0]-points[2][0]))
    b = ((points[0][0]**2)*(points[1][1]-points[2][1])+(points[2][0]**2)*(points[0][1]-points[1][1])+(points[1][0]**2)*(points[2][1]-points[0][1]))/((points[0][0]-points[1][0])*(points[0][0]-points[2][0])*(points[1][0]-points[2][0]))
    c = ((points[1][0]**2)*(points[2][0]*points[0][1]-points[0][0]*points[2][1]) + (points[1][0])*((points[0][0]**2)*points[2][1]-(points[2][0]**2)*points[0][1])+points[0][0]*points[2][0]*points[1][1]*(points[2][0]-points[0][0]))/((points[0][0]-points[1][0])*(points[0][0]-points[2][0])*(points[1][0]-points[2][0]))
    return a,b,c

#def get_kpts(atoms_obj):
#    structure = AseAtomAdaptor.get_structure(atoms_obj)
#    kpts_dat = get_kpoints(structure, minDistance = 30, include_gamma = False)
#    return kpts_dat['cords']

def main(metal:str,functional:str,slab_type:str, data_base:str, guess_lattice:float|None=None, grid_spacing:float=0.16):

    at_number = chemical_symbols.index(metal)
    functional_folder = sanitize(functional)
    folder_exist(functional_folder)
    folder_exist(f'{functional_folder}/{metal}_latt_fit')

    if guess_lattice is None:
        if slab_type != reference_states[at_number].get('symmetry'): raise ValueError('the given slab type does not match the saved type for ase guess lattice')
        guess_lattice = reference_states[at_number].get('a')

    parprint(f'lattice optimisation for {metal} with {functional}, guess latice is at {guess_lattice}')

    potential_energies = []
    lattices = [guess_lattice * 0.75,guess_lattice * 0.85, guess_lattice, guess_lattice * 1.15,guess_lattice * 1.25]
    for cur_lattice in lattices:

        #surface_builder = fcc100 if slab_type == 'fcc' else bcc100 if slab_type == 'bcc' else hcp0001 if slab_type == 'hcp' else None
        #bulk_con = surface_builder(symbol=metal, a=cur_lattice, size=(1, 1, 4),periodic=True)
        #bulk_con = Cluster(bulk_con)
        #bulk_con.minimal_box(border=[cur_lattice,cur_lattice,cur_lattice*4],h=grid_spacing,multiple=4)

        bulk_con = bulk(name=metal,crystalstructure=slab_type,a=cur_lattice)

        calc = GPAW(mode=PW(500),
                    xc=functional,
                    basis='dzp',
                    kpts = (10,10,10),#get_kpts(bulk_con),
                    txt=f'{functional_folder}/{metal}_latt_fit/lat-opt_{metal}_{slab_type}_a-{cur_lattice}.txt',
                    gpts=h2gpts(grid_spacing, bulk_con.get_cell(), idiv=4),
                    parallel={'augment_grids': True, 'sl_auto': True},
                    convergence={'eigenstates': 0.000001},
                    eigensolver=Davidson(3),
                    #hund=smile == 'O=O',
                    )

        bulk_con.set_calculator(calc)

        potential_energy = bulk_con.get_potential_energy()
        potential_energies.append(potential_energy)
        ### DELETE BULK AND CALC ###
        del calc
        del bulk_con

    parable = lambda x,a,b,c: a*x**2+b*x+c
    morse_potential = lambda x,D,a,r: D*(1-math.e**(-a*(x-r)))**2
    param,param_cov = curve_fit(morse_potential,lattices,potential_energies)

    #optimal_lattice = (-param[1])/(2*param[0]) # parable top point x = -b/2a
    optimal_lattice = param[-1]
    estimated_energy = morse_potential(optimal_lattice,*param)

    #surface_builder = fcc100 if slab_type == 'fcc' else bcc100 if slab_type == 'bcc' else hcp0001 if slab_type == 'hcp' else None
    #bulk_con = surface_builder(symbol=metal, a=optimal_lattice, size=(1, 1, 4))

    bulk_con = bulk(name=metal, crystalstructure=slab_type, a=optimal_lattice)

    calc = GPAW(mode=PW(500),
                xc=functional,
                basis='dzp',
                kpts=(10,10,10),#get_kpts(bulk_con),
                txt=f'{functional_folder}/lat-opt_{metal}_{slab_type}_a-{optimal_lattice}.txt',
                gpts=h2gpts(grid_spacing, bulk_con.get_cell(), idiv=4),
                parallel={'augment_grids': True, 'sl_auto': True},
                convergence={'eigenstates': 0.000001},
                eigensolver=Davidson(3),
                # hund=smile == 'O=O',
                )

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
        with db.connect(data_base) as db_obj:
            db_obj.write(bulk_con, functional=functional, lattice=optimal_lattice, data={'lattice_test':{'a':lattices, 'energies':potential_energies, 'parabola':param, 'parabola_covariance':param_cov}})



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('metal',type=str)
    parser.add_argument('surface_type',type=str,choices=('fcc','bcc','hcp'))
    parser.add_argument('func',type=str)
    parser.add_argument('--lattice','-a',type=float)
    parser.add_argument('-db','--database',help='directory to the database, if not stated will look for molreact.db in pwd.', default='slab_lattice_opt.db')
    args = parser.parse_args()

    main(metal=args.metal,functional=args.func,slab_type=args.surface_type,guess_lattice=args.lattice, data_base=args.database)


