import argparse
import math
#import os
#import re
import sys
import pathlib
from itertools import chain
#from dataclasses import dataclass, field
from typing import Sequence, NoReturn, Tuple, Iterable, Optional, NamedTuple
#from collections import namedtuple
from operator import itemgetter

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from scripts_for_adsorbate_database import sanitize, folder_exist, build_pd
from scripts_for_adsorbate_database.adsorbate_correlation_plot import Functional, component, adsorbate_reaction

#import ase.db as db
#from ase.db.core import bytes_to_object
#import matplotlib.pyplot as plt
#import numpy as np
#import pandas as pd
#import plotly.express as px
import plotly.graph_objects as go


def pearson(point_seq: Sequence[tuple[float, float]]) -> float:
    x_avg = sum(map(itemgetter(0), point_seq))/(n := len(point_seq))
    y_avg = sum(map(itemgetter(1), point_seq))/n
    return sum((x-x_avg)*(y-y_avg) for x, y in point_seq)/(math.sqrt(sum((x-x_avg)**2 for x, _ in point_seq))*math.sqrt(sum((y-y_avg)**2 for _, y in point_seq)))


def plot_correlation_matrix(reaction_seq: Sequence[adsorbate_reaction], BEEF_vdW_functional: Functional, png_bool: bool = False):
    fig = go.Figure()

    correlation_matrix = [[
        pearson(tuple(zip(BEEF_vdW_functional.calculate_BEE_reaction_enthalpy(reaction_1),
                          BEEF_vdW_functional.calculate_BEE_reaction_enthalpy(reaction_2))))
        for j, reaction_2 in enumerate(reaction_seq, start=i+1)] + [None]*(len(reaction_seq)-i)
        for i, reaction_1 in enumerate(reaction_seq)]

    text_matrix = [[
        f'{str(reaction_1)}<br>{str(reaction_2)}'
        for j, reaction_2 in enumerate(reaction_seq, start=i+1)] + [None]*(len(reaction_seq)-i)
        for i, reaction_1 in enumerate(reaction_seq)]

    fig.add_trace(go.Heatmap(
        z=correlation_matrix,
        x=(nr_axis := [str(i) for i in range(len(reaction_seq))]),
        y=nr_axis,
        text=text_matrix,
        texttemplate='{text}',
        hoverongaps=False
    ))

    folder_exist('reaction_plots')
    save_name = 'reaction_plots/' + f'correlation_matrix'
    if png_bool: fig.write_image(save_name + '.png')
    fig.write_html(save_name + '.html', include_mathjax='cdn')


def main(slab_db_dir: list[str], adsorbate_db_dir: list[str], mol_db_dir: list[str], png_bool: bool = False):
    pd_adsorbate_dat = build_pd(adsorbate_db_dir)
    pd_slab_dat = build_pd(slab_db_dir)
    pd_mol_dat = build_pd(mol_db_dir)

    functional_set = {'BEEF-vdW'} # {xc for _, row in pd_adsorbate_dat.iterrows() if not pd.isna((xc := row.get('xc')))}

    reactions = tuple(chain(*((
        adsorbate_reaction((('molecule', 'O=O', 0.5), ('molecule', '[HH]', 0.5), ('slab', f'{metal}_111', 1)), (('adsorbate', f'{metal}_111_OH_top', 1),)),  # 0
        adsorbate_reaction((('molecule', 'O=O', 1), ('molecule', '[HH]', 0.5), ('slab', f'{metal}_111', 1)), (('adsorbate', f'{metal}_111_OOH_top', 1),)),  # 1
        adsorbate_reaction((('molecule', 'O', 2), ('slab', f'{metal}_111', 1)), (('adsorbate', f'{metal}_111_OOH_top', 1), ('molecule', '[HH]', 1.5))),  # 2
        adsorbate_reaction((('molecule', 'O', 1), ('slab', f'{metal}_111', 1)), (('adsorbate', f'{metal}_111_OH_top', 1), ('molecule', '[HH]', 0.5))),  # 3
        adsorbate_reaction((('molecule', 'OO', 1), ('slab', f'{metal}_111', 1)), (('adsorbate', f'{metal}_111_OOH_top', 1), ('molecule', '[HH]', 0.5))),  # 4
        adsorbate_reaction((('molecule', 'OO', 0.5), ('slab', f'{metal}_111', 1)), (('adsorbate', f'{metal}_111_OH_top', 1),)),  # 5
                        )for metal in ['Pt', 'Cu', 'Pd', 'Rh'])))

    metal_ref_ractions = tuple(chain(*((
        adsorbate_reaction((('adsorbate', 'Pt_111_OH_top', 1), ('slab', f'{metal}_111', 1)), (('adsorbate', f'{metal}_111_OH_top', 1), ('slab', 'Pt_111', 1))), #8
        adsorbate_reaction((('adsorbate', 'Pt_111_OOH_top', 1), ('slab', f'{metal}_111', 1)), (('adsorbate', f'{metal}_111_OOH_top', 1), ('slab', 'Pt_111', 1))), #9
                        )for metal in ['Cu', 'Pd', 'Rh'])))

    all_reactions = reactions + metal_ref_ractions

    dictionary_of_needed_strucs = {'molecule': [], 'slab': [], 'adsorbate': []}
    for reac in all_reactions:
        for compo in reac.reactants + reac.products:
            dictionary_of_needed_strucs[compo.type].append(compo.name)

    functional_list = [Functional(functional_name='BEEF-vdW', slab_db=pd_slab_dat, adsorbate_db=pd_adsorbate_dat, mol_db=pd_mol_dat, needed_struc_dict=dictionary_of_needed_strucs, thermo_dynamic=False)]

    plot_correlation_matrix(all_reactions, functional_list[0], png_bool=png_bool)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-adb', '--adsorbate_db', nargs='+', help='path to one or more databases containing the data.')
    parser.add_argument('-mdb', '--molecule_db', nargs='+', help='path to one or more databases containing the data.')
    parser.add_argument('-sdb', '--slab_db', nargs='+', help='path to one or more databases containing the data.')
    #parser.add_argument('-m', '--metals', nargs='+', default=['Pt', 'Cu'])
    parser.add_argument('-png', '--png', action='store_true', default=False,)
    args = parser.parse_args()

    main(slab_db_dir=args.slab_db, adsorbate_db_dir=args.adsorbate_db, mol_db_dir=args.molecule_db, png_bool=args.png)
