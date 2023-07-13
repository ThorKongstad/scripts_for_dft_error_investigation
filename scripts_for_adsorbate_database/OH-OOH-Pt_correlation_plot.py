import argparse
import os
import re
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Sequence, NoReturn, Tuple, Iterable, Optional, NamedTuple
#from collections import namedtuple

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from scripts_for_adsorbate_database import sanitize, folder_exist, build_pd

#import ase.db as db
from ase.db.core import bytes_to_object
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import plotly.express as px
import plotly.graph_objects as go


class component(NamedTuple):
    type: str
    name: str
    amount: float


@dataclass
class adsorbate_reaction:
    reactants: Sequence[Tuple[str, str, float] | component]
    products: Sequence[Tuple[str, str, float] | component]

    def __post_init__(self):
        #component = namedtuple('component', ['type', 'name', 'amount'])
        for n, reacs_or_prods in enumerate([self.reactants, self.products]):
            new_component_seq = []
            for i, reac_or_prod in enumerate(reacs_or_prods):
                if len(reac_or_prod) == 3: raise ValueError('a component of a reaction does not have the correct size')
                if not reac_or_prod in ('molecule', 'slab', 'adsorbate'): raise ValueError('The reactant or product type sting appear to be wrong')
                new_component_seq.append(component(*reac_or_prod) if not isinstance(reac_or_prod, component) else reac_or_prod)
            setattr(self, 'reactants' if n == 0 else 'products', tuple(new_component_seq))

    def __str__(self):
        return ' ---> '.join([' + '.join([f'{reac.amount:.2g}{reac.name if reac.name != "cid281" else "C|||O"}({reac.type})' for reac in comp]) for comp in (self.reactants,self.products)])


class Functional:
    def __init__(self, functional_name: str, slab_db: pd.DataFrame, adsorbate_db: pd.DataFrame, mol_db: pd.DataFrame, needed_struc_dict: Optional[dict[str, list[str]]] = None):
        self.name = functional_name
        self.molecule = {smile: mol_db.query(f'smiles == "{smile}" and xc == "{functional_name}" and enthalpy.notna()').get('enthalpy').iloc[0] for smile in needed_struc_dict['molecule']}
        self.slab = {structure_str: slab_db.query(f'structure_str == "{structure_str}" and xc == "{functional_name}" and energy.notna()').get('energy').iloc[0] for structure_str in needed_struc_dict['slabs']}
        self.adsorbate = {structure_str: adsorbate_db.query(f'structure_str == "{structure_str}" and xc == "{functional_name}" and enthalpy.notna()').get('enthalpy').iloc[0] for structure_str in needed_struc_dict['adsorbate']}

        self.has_BEE = functional_name == 'BEEF-vdW'
        if self.has_BEE:
            self.molecule_energy = {smile: mol_db.query(f'smiles == "{smile}" and xc == "{functional_name}" and enthalpy.notna()').get('energy').iloc[0] for smile in needed_struc_dict['molecule']}
            self.slab_energy = self.slab
            self.adsorbate_energy = {structure_str: adsorbate_db.query(f'structure_str == "{structure_str}" and xc == "{functional_name}" and enthalpy.notna()').get('energy').iloc[0] for structure_str in needed_struc_dict['adsorbate']}
            self.molecule_bee = {smile: np.array(bytes_to_object(mol_db.query(f'smiles == "{smile}" and xc == "{functional_name}" and enthalpy.notna()').get('_data').iloc[0]).get('ensemble_en'))[:] for smile in needed_struc_dict['molecule']}
            self.slab_bee = {structure_str: np.array(bytes_to_object(mol_db.query(f'structure_str == "{structure_str}" and xc == "{functional_name}" and enthalpy.notna()').get('_data').iloc[0]).get('ensemble_en'))[:] for structure_str in needed_struc_dict['slabs']}
            self.adsorbate_bee = {structure_str: np.array(bytes_to_object(mol_db.query(f'structure_str == "{structure_str}" and xc == "{functional_name}" and enthalpy.notna()').get('_data').iloc[0]).get('ensemble_en'))[:] for structure_str in needed_struc_dict['adsorbate']}
        else:
            self.molecule_energy = {}
            self.adsorbate_energy ={}
            self.slab_energy = {}
            self.molecule_bee = {}
            self.slab_bee = {}
            self.adsorbate_bee = {}

    def calculate_reaction_enthalpy(self, reaction: adsorbate_reaction) -> float:
        reactant_enthalpy, product_enthalpy = tuple(sum(getattr(self, typ)[name] * amount for typ, name, amount in getattr(reaction, reac_part)) for reac_part in ('reactants', 'products'))
        return product_enthalpy - reactant_enthalpy

    def calculate_reaction_energy(self, reaction: adsorbate_reaction) -> float:
        if not self.has_BEE: raise ValueError('calculate_reaction_energy only if the functional has BEE')
        reactant_enthalpy, product_enthalpy = tuple(sum(getattr(self, typ + '_energy')[name] * amount for typ, name, amount in getattr(reaction, reac_part)) for reac_part in ('reactants', 'products'))
        return product_enthalpy - reactant_enthalpy

    def calculate_BEE_reaction_enthalpy(self, reaction: adsorbate_reaction) -> np.ndarray[float]:
        if not self.has_BEE: raise ValueError('calculate_reaction_energy only if the functional has BEE')
        correction = self.calculate_reaction_enthalpy(reaction) - self.calculate_reaction_energy(reaction)
        reactant_BEE_enthalpy, product_BEE_enthalpy = tuple(sum(getattr(self, typ + '_bee')[name] * amount for typ, name, amount in getattr(reaction, reac_part)) for reac_part in ('reactants', 'products'))
        return product_BEE_enthalpy - reactant_BEE_enthalpy + correction


def correlation_plotly(reaction_1: adsorbate_reaction, reaction_2: adsorbate_reaction, functional_seq: Sequence[Functional], reaction_indexes: Optional[Tuple[int, int]] = None):
    fig = go.Figure()

    colour_dict = {
        'PBE': 'idianred',
        'RPBE': 'firebrick',
        'PBE-PZ-SIC': 'darkorange',
        'BEEF-vdW': 'mediumblue',
        "{'name':'BEEF-vdW','backend':'libvdwxc'}": 'mediumpurple'
    }

    for c_nr, func in enumerate(functional_seq):
        marker_arg = dict(marker={'color': colour_dict[func.name]}) if func in colour_dict.keys() else {}
        try:
            fig.add_trace(go.Scatter(
             x=(func.calculate_reaction_enthalpy(reaction_1),),
             y=(func.calculate_reaction_enthalpy(reaction_2),),
             name=func.name,
             mode='markers',
             **marker_arg))
            if func.name == 'BEEF-vdW':
                try:
                    ensamble_trace = go.Scatter(
                        x=func.calculate_BEE_reaction_enthalpy(reaction_1).tolist(),
                        y=func.calculate_BEE_reaction_enthalpy(reaction_2).tolist(),
                        name=f'BEE for {func}',
                        mode='markers',
                        marker=dict(color='Grey',opacity=0.5,)
                    )
                    fig.data = (ensamble_trace,) + fig.data
                except: pass
        except: continue

    fig.update_layout(
        xaxis_title=str(reaction_1),
        yaxis_title=str(reaction_2)
    )

    folder_exist('reaction_plots')
    if reaction_indexes: save_name = 'reaction_plots/' + f'correlation_plot_{"-".join([str(x) for x in reaction_indexes])}.html'
    else: save_name = 'reaction_plots/correlation_plot.html'
    fig.write_html(save_name, include_mathjax='cdn')


def main(reaction_index_1, reaction_index_2, slab_db_dir: list[str], adsorbate_db_dir: list[str], mol_db_dir: list[str]):
    pd_adsorbate_dat = build_pd(adsorbate_db_dir)
    pd_slab_dat = build_pd(slab_db_dir)
    pd_mol_dat = build_pd(mol_db_dir)

    functional_set = {xc for _, row in pd_adsorbate_dat.iterrows() if not pd.isna((xc := row.get('xc')))}

    reactions = (
        adsorbate_reaction((('molecule', 'O=O', 0.5), ('molecule', '[HH]', 0.5), ('slab', 'Pt_111', 1)), (('adsorbate', 'Pt_111_OH_top', 1),)),
        adsorbate_reaction((('molecule', 'O=O', 1), ('molecule', '[HH]', 0.5), ('slab', 'Pt_111', 1)), (('adsorbate', 'Pt_111_OOH_top', 1),)),
    )

    dictionary_of_needed_strucs = {'molecule': [], 'slab': [], 'adsorbate': []}
    for reac in reactions:
        for compo in reac.reactants + reac.products:
            dictionary_of_needed_strucs[compo.type].append(compo.name)

    functional_list = [Functional(functional_name=xc, slab_db=pd_slab_dat, adsorbate_db=pd_adsorbate_dat, mol_db=pd_mol_dat, needed_struc_dict=dictionary_of_needed_strucs) for xc in functional_set]

    correlation_plotly(reaction_1=reactions[reaction_index_1], reaction_2=reactions[reaction_index_2], functional_seq=functional_list, reaction_indexes=(reaction_index_1, reaction_index_2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('reaction_1',type=int)
    parser.add_argument('reaction_2',type=int)
    parser.add_argument('-adb', '--adsorbate_db', nargs='+', help='path to one or more databases containing the data.')
    parser.add_argument('-mdb', '--molecule_db', nargs='+', help='path to one or more databases containing the data.')
    parser.add_argument('-sdb', '--slab_db', nargs='+', help='path to one or more databases containing the data.')
    args = parser.parse_args()

    main(reaction_index_1=args.reaction_1, reaction_index_2=args.reaction_2, slab_db_dir=args.slab_db, adsorbate_db_dir=args.adsorbate_db, mol_db_dir=args.molecule_db)
