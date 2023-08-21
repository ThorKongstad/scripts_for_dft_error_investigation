import argparse
import os
import re
from ase.visualize import view
from ase import Atoms
from ase.data.pubchem import pubchem_atoms_search
from ase.build import molecule
from ase.constraints import FixAtoms
import ase.db as db
from ase.build import fcc100,fcc110,fcc111,fcc211,bcc100,bcc110,bcc111, hcp0001, add_adsorbate
from gpaw.cluster import Cluster
from typing import NoReturn,Tuple
from collections import namedtuple

def folder_exist(folder_name: str) -> NoReturn:
    if folder_name not in os.listdir(): os.mkdir(folder_name)


def sanitize(unclean_str: str) -> str:
    for ch in ['!', '*', '?', '{', '[', '(', ')', ']', '}', "'",'"']: unclean_str = unclean_str.replace(ch, '')
    for ch in ['/', '\\', '|',' ']: unclean_str = unclean_str.replace(ch, '_')
    for ch in ['=', '+', ':']: unclean_str = unclean_str.replace(ch, '-')
    return unclean_str

def facet_tuple_parser(metal:str,facet:Tuple[str,str], size = (2,2,4),orthogonal:bool=None,latice_constant:None|float=None) -> Atoms:
    lat_dict = {'a':latice_constant} if latice_constant is not None else {}
    orth_dict = {'orthogonal':orthogonal} if orthogonal is not None else {}
    if facet[0].lower() == 'fcc':
        if facet[1] == '100': surface = fcc100(symbol=metal,size=size,**lat_dict)
        elif facet[1] == '110': surface = fcc110(symbol=metal,size=size,**lat_dict)
        elif facet[1] == '111': surface = fcc111(symbol=metal,size=size,**lat_dict,**orth_dict)
        elif facet[1] == '211': surface = fcc211(symbol=metal, size=size, **lat_dict, **orth_dict)
        else: raise 'facet is not implemented for fcc'
    elif facet[0].lower() == 'bcc':
        if facet[1] == '100': surface = bcc100(symbol=metal,size=size,**lat_dict)
        elif facet[1] == '110': surface = bcc110(symbol=metal,size=size,**lat_dict,**orth_dict)
        elif facet[1] == '110': surface = bcc111(symbol=metal,size=size,**lat_dict,**orth_dict)
        else: raise 'facet is not implemented for bcc'
    elif facet[0].lower() == 'hcp':
        if facet[1] == '0001': surface = hcp0001(symbol=metal,size=size,**lat_dict,**orth_dict)
        else: raise 'facet is not implemented for hcp'
    else: raise 'type of crystal not recognised'
    return surface

def get_mol(smiles_cid:str) -> Atoms:
    if smiles_cid == '[HH]': atoms = Cluster(molecule('H2'))  # pubchem search hates hydrogen, it hates its name, it hates its cid and most of all it hates its weird smile and dont you dare confuse HH for hydrogen
    elif 'cid' in smiles_cid: atoms = Cluster(pubchem_atoms_search(cid=int(smiles_cid.replace('cid',''))))
    else: atoms = Cluster(pubchem_atoms_search(smiles=smiles_cid))
    return atoms


def main(metal:str, facet:Tuple[str,str], functional,
         orthogonal:bool|None=None, lattice_constant: None | float=None, size: Tuple[float,float,float] = (2, 2, 4), adsorbate: str | None=None, adsorbate_placement: str | None=None,rotation:Tuple[str,float]|None=None, grid_spacing:float=0.16,view_bool:bool=False):
    slab = facet_tuple_parser(metal, facet, size=size, orthogonal=orthogonal, latice_constant=lattice_constant)

    slab.set_constraint(constraint=FixAtoms(mask=[at.tag in (1,2,3) for at in slab]))

    if adsorbate:
        if not adsorbate_placement: raise ValueError('adsorbate placement must be stated together with an adsorbate.')
        mol = get_mol(adsorbate)
        if rotation:
            for ax in rotation[0]:
                mol.rotate(a=rotation[1],v=ax,center='COP')
        add_adsorbate(slab,mol,2,position=adsorbate_placement,mol_index=mol.positions.tolist().index(min(mol.positions.tolist(),key=lambda x:x[-1])))
    slab = Cluster(slab)
    slab.minimal_box(border=[0,0,6],h=grid_spacing,multiple=4)

    if view_bool:view(slab)
    else:
        with db.connect('data_base_place_holder') as db_obj:
            db_obj.write(atoms=slab, xc=functional, lattice=lattice_constant, adsorbate=adsorbate, adsorbate_placement=adsorbate_placement, relaxed=False, vibration=False, grid_spacing=grid_spacing)



if __name__ == '__main__':
    class ValidateFacet(argparse.Action):
        def __call__(self,parser, args, values, option_string=None, **kwargs):
            valid_crystals = ('fcc','bcc','hcp') # list of valid values for the first arg known as crystal
            valid_facets = ('100','110','111','211','0001') # list of valid values for the second arg known as facet
            crystal,facet = values
            if crystal.lower() not in valid_crystals: raise ValueError('unknown crystal structure')
            if facet not in valid_facets: raise ValueError('unknown facet structure')
            facet_dat = namedtuple('facet_dat','crystal facet')
            setattr(args,self.dest,facet_dat(crystal,facet))

    class ValidateAdsorbate(argparse.Action):
        def __call__(self, parser, args, values, option_string=None, **kwargs):
            valid_sites = ('ontop', 'bridge', 'hollow', 'fcc', 'hcp', 'longbridge', 'shortbridge')
            site,adsorbate = values
            if site not in valid_sites: raise ValueError('unknown adsorbate site')
            Adsorbate_dat = namedtuple('Adsorbate_dat', 'site adsorbate')
            setattr(args, self.dest, Adsorbate_dat(site, adsorbate))

    class ValidateRotation(argparse.Action):
        def __call__(self, parser, args, values, option_string=None, **kwargs):
            axis,rotation = values
            if not all(a in 'xyz' for a in axis): raise ValueError('axis of rotation must be a combination of xyz')
            if not re.match('-?\d+(.\d+)?',rotation): raise ValueError('rotation input must be a number.')
            rotation_dat = namedtuple('rotation_dat','axis degress')
            setattr(args, self.dest, rotation_dat(axis,float(rotation)))


    parser = argparse.ArgumentParser()
    parser.add_argument('metal',help='str denoting for metal to make up the surface')
    parser.add_argument('facet',action=ValidateFacet,nargs=2,help='2 inputs must be given like "fcc 100 ", the first is the type of crystal and the second is the miller indici of the surface')
    parser.add_argument('functional',help='str denoting what fucntional to calculate with')
    parser.add_argument('--latice_constant','-a', type=float, help='latice constant of the metal, if nothing is given it will take the ase default which is given from experiment')
    parser.add_argument('--size', '-s', type=int, nargs=3, default=(2, 2, 4), help='3 numbers for the size given as x y z. standard is 2 2 4. note size is not atom count in x distance, it is minimum unit cell in Å. would have been alot better to just have size * a = unitcell length')
    parser.add_argument('--adsorbate','-ad', default=(None,None), action=ValidateAdsorbate, nargs=2, help='2 inputs must be given, the first a placement on the surface and the smiles denoting adsorbate')
    parser.add_argument('--rotate','-r',action=ValidateRotation,nargs=2,help='defines a rotation for the adsorbate, 2 nargs; the first a string og the axes given in a combinatio of xyz, the second the degrees that each axis should turn, so far only one value for all given axis is implemented.')
    parser.add_argument('--grid_spacing', '-g', type=int, help='vacum grid spacing, default is 0.16', default=0.16)
    parser.add_argument('--orthogonal','-ort',default=None,action='store_true', help='states whether the orthogonal key=True should be given to the builder')
    parser.add_argument('--view','-view',action='store_true',help='if stated ase gui will be opened instead of placing the structure in the database.')
    args=parser.parse_args()

    main(
         args.metal,facet=args.facet,functional=args.functional,lattice_constant=args.latice_constant, size=args.size,
         orthogonal=args.orthogonal,adsorbate_placement=args.adsorbate[0],adsorbate=args.adsorbate[1],
         rotation=args.rotate, grid_spacing=args.grid_spacing,view_bool=args.view
    )