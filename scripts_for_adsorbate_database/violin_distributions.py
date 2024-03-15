import argparse
from itertools import chain
import math
import sys
import pathlib
from typing import Sequence, Optional, Iterable
import traceback
from re import match
from dataclasses import dataclass, field

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from scripts_for_adsorbate_database import sanitize, folder_exist, build_pd, adsorbate_reaction, adsorption_OH_reactions, adsorption_OOH_reactions, metal_ref_ractions, sd, mean
from scripts_for_adsorbate_database.adsorbate_correlation_plot import Functional

import numpy as np
from scipy import stats, odr
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def one_dim_violin(functional_list: Sequence[Functional], oh_reactions: Sequence[adsorbate_reaction], ooh_reactions: Sequence[adsorbate_reaction], save_name: str,  png_bool: bool = False):
    assert len(oh_reactions) == len(ooh_reactions)

    def single_plot(functional_list: Sequence[Functional], reaction: adsorbate_reaction) -> go.Figure:
        metal = reaction.products[0].name.split('_')[0]
        colour_dict_functional = {
            'PBE': '#CD5C5C',
            'RPBE': '#B22222',
            'PBE-PZ-SIC': '#FF8C00',
            'BEEF-vdW': '#0000CD',
            "{'name':'BEEF-vdW','backend':'libvdwxc'}": '#9370DB',
            'TPSS': px.colors.qualitative.Dark2[5]
        }

        marker_dict_functional = {
            'PBE': 'square',
            'RPBE': 'star-square',
            'PBE-PZ-SIC': '#FF8C00',
            'BEEF-vdW': 'circle',
            "{'name':'BEEF-vdW','backend':'libvdwxc'}": 'octagon',
            'TPSS': 'diamond'
        }

        colour_dict_metal = dict(
            Pt=px.colors.qualitative.Prism[1],
            Cu=px.colors.qualitative.Plotly[1],
            Pd=px.colors.qualitative.Safe[4],
            Rh=px.colors.qualitative.Vivid[5],
            Ag=px.colors.qualitative.Pastel[10],
            Ir=px.colors.qualitative.Dark2[7],
            Au=px.colors.qualitative.Dark2[5],
        )

        fig = go.Figure()
        for xc in functional_list:
            line_arg = dict(line=dict(color=colour_dict_functional[xc.name], )) if xc.name in colour_dict_functional.keys() else dict(line=dict(color='DarkSlateGrey'))
            marker_arg = dict(marker=dict(size=16, color=colour_dict_metal[metal] if metal in colour_dict_metal.keys() else 'DarkSlateGrey', symbol=marker_dict_functional[xc.name] if xc.name in marker_dict_functional.keys() else 'circle'))
            try: fig.add_trace(go.Scatter(
                mode='markers',
                name=f'{xc.name} {metal}',
                y=0,
                x=xc.calculate_reaction_enthalpy(reaction),
                hovertemplate='E_dft =  %{x:.3f} eV',
                legendgroup=metal,
                legendgrouptitle_text=metal,
                **marker_arg
                ))
            except: traceback.print_exc()
            if xc.has_BEE:
                try:
                    #fig.add_trace(go.Scatter(
                        #mode='markers',
                        #name=f'BEE for {metal} {xc.name}',
                        #y=0,
                        #x=(ens_x_cloud := xc.calculate_BEE_reaction_enthalpy(reaction).tolist()),
                       # marker=dict(color=colour_dict_metal[metal] if metal in colour_dict_metal.keys() else 'Grey', opacity=0.5),
                      #  legendgroup=metal,
                     #   legendgrouptitle_text=metal,
                    #))
                    fig.add_trace(go.Violin(
                        name=f'BEE for {metal} {xc.name} violin',
                        x=(ens_x_cloud := xc.calculate_BEE_reaction_enthalpy(reaction).tolist()),
                        orientation='h',
                        points='all',
                        hovertemplate=f'E_dft = {mean(ens_x_cloud)} +- {(err := sd(ens_x_cloud))}',
                        **marker_arg,
                        **line_arg
                    ))
                    fig.update_traces(selector=dict(name=f'{xc.name} {metal}'),
                                      error_y=dict(type='constant', value=0, color=colour_dict_metal[metal] if metal in colour_dict_metal.keys() else 'Grey', thickness=1.5, width=3, visible=False),
                                      error_x=dict(type='constant', value=err, color=colour_dict_metal[metal] if metal in colour_dict_metal.keys() else 'Grey', thickness=1.5, width=3, visible=False),)
                    fig.data = fig.data[-1:] + fig.data[0:-1]
                except: traceback.print_exc()
        return fig


    largest_err_axis = {'axis': False, 'err': 0}
    fig = make_subplots(rows=len(oh_reactions), cols=2)
    for i, (oh_reac, ooh_reac) in enumerate(zip(oh_reactions, ooh_reactions)):
        for j, reac in enumerate((oh_reac, ooh_reac)):
            sub_fig = single_plot(functional_list, reac)
            fig.add_trace(sub_fig.data, col=j + 1, row=i + 1)
            fig.update_layout(title_text=str(reac), xaxis_title='eV', col=j + 1, row=i + 1)
            fig.update_yaxes(visible=False, col=j + 1, row=i + 1)

            try:
                if largest_err_axis['err'] < (err := [tra['error_x']['value'] for tra in sub_fig.data if 'error_x' in dict(tra).keys()][0]):
                    largest_err_axis = {'axis': f'x{i+1+j}', 'err': err}
            except: pass

    if largest_err_axis['axis']:
        for i in range(len(oh_reactions)):
            for j in range(2):
                if largest_err_axis['axis'] == f'x{i+1+j}': continue
                fig.update_xaxes(
                    scaleanchor=largest_err_axis['axis'],
                    scaleratio=1,
                    col=j + 1, row=i + 1
                )

    folder_exist('reaction_plots')
    if png_bool: fig.write_image(save_name + '.png')
    fig.write_html(save_name + '.html', include_mathjax='cdn')



def main(slab_db_dir: list[str], adsorbate_db_dir: list[str], mol_db_dir: list[str], thermo_dynamics: bool = False):
    pd_adsorbate_dat = build_pd(adsorbate_db_dir)
    pd_slab_dat = build_pd(slab_db_dir)
    pd_mol_dat = build_pd(mol_db_dir)

    functional_set = {xc for _, row in pd_adsorbate_dat.iterrows() if not pd.isna((xc := row.get('xc')))}

    oh_ad_h2_water = adsorption_OH_reactions[1::3]  # metal_ref_ractions[0::2] #adsorption_OH_reactions[1::3] #[1,4,7,10,13,16]
    ooh_ad_h2_water = adsorption_OOH_reactions[1::3]  # metal_ref_ractions[1::2] #adsorption_OOH_reactions[1::3]

    oh_ad_h2_per_ox = adsorption_OH_reactions[2::3]  # metal_ref_ractions[0::2] #adsorption_OH_reactions[1::3] #[1,4,7,10,13,16]
    ooh_ad_h2_per_ox = adsorption_OOH_reactions[2::3]  # metal_ref_ractions[1::2] #adsorption_OOH_reactions[1::3]

    oh_ad_h2_ox = adsorption_OH_reactions[0::3]  # metal_ref_ractions[0::2] #adsorption_OH_reactions[1::3] #[1,4,7,10,13,16]
    ooh_ad_h2_ox = adsorption_OOH_reactions[0::3]  # metal_ref_ractions[1::2] #adsorption_OOH_reactions[1::3]

    oh_ad_metal_ref = metal_ref_ractions[0::3] #adsorption_OH_reactions[1::3] #[1,4,7,10,13,16]
    ooh_ad_metal_ref = metal_ref_ractions[1::3] #adsorption_OOH_reactions[1::3]

    dictionary_of_needed_strucs = {'molecule': [], 'slab': [], 'adsorbate': []}
    all_reaction = chain(
        oh_ad_h2_water, ooh_ad_h2_water,
                oh_ad_h2_per_ox, ooh_ad_h2_per_ox,
                oh_ad_h2_ox, ooh_ad_h2_ox,
                oh_ad_metal_ref, ooh_ad_metal_ref
                )

    for reac in all_reaction:
        for compo in reac.reactants + reac.products:
            if compo.name not in dictionary_of_needed_strucs[compo.type]:
                dictionary_of_needed_strucs[compo.type].append(compo.name)

    functional_list = []
    for xc in functional_set:
        try: functional_list.append(Functional(functional_name=xc, slab_db=pd_slab_dat, adsorbate_db=pd_adsorbate_dat, mol_db=pd_mol_dat, needed_struc_dict=dictionary_of_needed_strucs, thermo_dynamic=thermo_dynamics))
        except: pass

    reactions_groups = (
        ('violins_water_ref', oh_ad_h2_water, ooh_ad_h2_water),
        ('violins_per_oxide_ref', oh_ad_h2_per_ox, ooh_ad_h2_per_ox),
        ('violins_oxigen_reg', oh_ad_h2_ox, ooh_ad_h2_ox),
        ('violins_metal_ref', oh_ad_metal_ref, ooh_ad_metal_ref)
    )

    for name, oh_reacs, ooh_reacs in reactions_groups:
        one_dim_violin(functional_list, oh_reacs, ooh_reacs, name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-adb', '--adsorbate_db', nargs='+', help='path to one or more databases containing the data.')
    parser.add_argument('-mdb', '--molecule_db', nargs='+', help='path to one or more databases containing the data.')
    parser.add_argument('-sdb', '--slab_db', nargs='+', help='path to one or more databases containing the data.')
    args = parser.parse_args()

    main(slab_db_dir=args.slab_db, adsorbate_db_dir=args.adsorbate_db, mol_db_dir=args.molecule_db)
