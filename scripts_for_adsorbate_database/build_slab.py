import argparse
#import os
#import sys
#import pathlib
from typing import NoReturn, Tuple, Optional
from collections import namedtuple
from ase.io import write
from ase.visualize import view
from ase import Atoms
from ase.build import fcc100, fcc110, fcc111, fcc211, bcc100, bcc110, bcc111, hcp0001

#sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


def facet_tuple_parser(metal: str, facet: Tuple[str, str], size=(2, 2, 4), orthogonal: bool = None, lattice_constant: Optional[float] = None) -> Atoms:
    lat_dict = {'a': lattice_constant} if lattice_constant is not None else {}
    orth_dict = {'orthogonal': orthogonal} if orthogonal is not None else {}
    if facet[0].lower() == 'fcc':
        if facet[1] == '100': surface = fcc100(symbol=metal, size=size, **lat_dict)
        elif facet[1] == '110': surface = fcc110(symbol=metal, size=size, **lat_dict)
        elif facet[1] == '111': surface = fcc111(symbol=metal, size=size, **lat_dict, **orth_dict)
        elif facet[1] == '211': surface = fcc211(symbol=metal, size=size, **lat_dict, **orth_dict)
        else: raise 'facet is not implemented for fcc'
    elif facet[0].lower() == 'bcc':
        if facet[1] == '100': surface = bcc100(symbol=metal, size=size, **lat_dict)
        elif facet[1] == '110': surface = bcc110(symbol=metal, size=size, **lat_dict, **orth_dict)
        elif facet[1] == '110': surface = bcc111(symbol=metal, size=size, **lat_dict, **orth_dict)
        else: raise 'facet is not implemented for bcc'
    elif facet[0].lower() == 'hcp':
        if facet[1] == '0001': surface = hcp0001(symbol=metal, size=size, **lat_dict, **orth_dict)
        else: raise 'facet is not implemented for hcp'
    else: raise 'type of crystal not recognised'
    return surface


def main(metal: str, facet: Tuple[str, str], orthogonal: Optional[bool] = None, lattice_constant: Optional[float] = None, size: Tuple[float, float, float] = (2, 2, 4), vacuum: float = 10, view_bool: bool = False):
    slab = facet_tuple_parser(metal, facet, size=size, orthogonal=orthogonal, lattice_constant=lattice_constant)
    slab.center(vacuum=vacuum, axis=2)

    if view_bool: view(slab)
    else: write(f'{metal}_{facet}.traj', slab)


if __name__ == '__main__':
    class ValidateFacet(argparse.Action):
        def __call__(self,parser, args, values, option_string=None, **kwargs):
            valid_crystals = ('fcc', 'bcc', 'hcp') # list of valid values for the first arg known as crystal
            valid_facets = ('100', '110', '111', '211', '0001') # list of valid values for the second arg known as facet
            crystal, facet = values
            if crystal.lower() not in valid_crystals: raise ValueError('unknown crystal structure')
            if facet not in valid_facets: raise ValueError('unknown facet structure')
            facet_dat = namedtuple('facet_dat', 'crystal facet')
            setattr(args, self.dest, facet_dat(crystal, facet))

    parser = argparse.ArgumentParser()
    parser.add_argument('metal', help='str denoting for metal to make up the surface')
    parser.add_argument('facet', action=ValidateFacet, nargs=2, help='2 inputs must be given like "fcc 100 ", the first is the type of crystal and the second is the miller indici of the surface')
    parser.add_argument('--lattice_constant', '-a', type=float, help='latice constant of the metal, if nothing is given it will take the ase default which is given from experiment')
    parser.add_argument('--size', '-s', type=int, nargs=3, default=(2, 2, 4), help='3 numbers for the size given as x y z. standard is 2 2 4. note size is not atom count in x distance, it is minimum unit cell in Å. would have been alot better to just have size * a = unitcell length')
    parser.add_argument('--vacuum', '-vac', type=float, default=10, help='Denotes the size of the vacuum around the slab, default is 10 Å')
    parser.add_argument('--orthogonal', '-ort', default=None, action='store_true', help='states whether the orthogonal key=True should be given to the builder')
    parser.add_argument('--view', '-view', action='store_true', help='if stated ase gui will be opened instead of placing the structure in the database.')
    args = parser.parse_args()

    main(
        metal=args.metal,
        facet=args.facet,
        lattice_constant=args.lattice_constant,
        size=args.size,
        vacuum=args.vacuum,
        orthogonal=args.orthogonal,
        view_bool=args.view
    )