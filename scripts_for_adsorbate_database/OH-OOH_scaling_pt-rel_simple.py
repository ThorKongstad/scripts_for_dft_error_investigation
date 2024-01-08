import argparse
import math
import sys
import pathlib
from typing import Sequence, Optional
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from scripts_for_adsorbate_database import sanitize, folder_exist, build_pd, adsorbate_reaction, adsorption_OH_reactions, adsorption_OOH_reactions, metal_ref_ractions
from scripts_for_adsorbate_database.adsorbate_correlation_plot import Functional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def scaling_plot(functional_list: Sequence[Functional], oh_reactions: Sequence[adsorbate_reaction], ooh_reactions: Sequence[adsorbate_reaction], png_bool: bool = False):
    fig = go.Figure()

    colour_dict_functional = {
        'PBE': 'indianred',
        'RPBE': 'firebrick',
        'PBE-PZ-SIC': 'darkorange',
        'BEEF-vdW': 'mediumblue',
        "{'name':'BEEF-vdW','backend':'libvdwxc'}": 'mediumpurple',
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

    #line = np.linspace(0, 2, 500)

    #fig.add_trace(go.Scatter(
    #    mode='lines',
    #    x=line,
    #    y=[3.2]*len(line),
    #    line=dict(
    #        color='Grey',
    #        #opacity=0.5
    #    ),
    #    showlegend=False,
    #))

    for oh_reac, ooh_reac in zip(oh_reactions, ooh_reactions):
        assert (metal := oh_reac.products[0].name.split('_')[0]) == ooh_reac.products[0].name.split('_')[0]
        marker_arg = dict(marker={'color': colour_dict_metal[metal], 'size': 16}) if metal in colour_dict_metal.keys() else dict(marker={'size': 16})
        for xc in functional_list:
            try: fig.add_trace(go.Scatter(
                mode='markers',
                name=f'{xc.name}-{metal}',
                x=[(oh_adsorp := xc.calculate_reaction_enthalpy(oh_reac))],
                y=[xc.calculate_reaction_enthalpy(ooh_reac)],
                hovertemplate=f'metal: {metal}' + '<br>' + f'XC: {xc.name}' + '<br>' + f'OH adsorption: {str(oh_reac)}' + '   %{x:.3f}' + '<br>' + f'OOH adsorption: {str(ooh_reac)}'+ '   %{y:.3f}',
                legendgroup=metal,
                legendgrouptitle_text=metal,
                **marker_arg
                ))
            except: traceback.print_exc()
            if xc.has_BEE:
                try:
                    fig.add_trace(go.Scatter(
                        mode='markers',
                        name=f'BEE for {metal}',
                        y=xc.calculate_BEE_reaction_enthalpy(ooh_reac).tolist(),
                        x=xc.calculate_BEE_reaction_enthalpy(oh_reac).tolist(),
                        hovertemplate=f'metal: {metal}' + '<br>' + f'OH adsorption: {str(oh_reac)}' + '   %{x:.3f}' + '<br>' + f'OOH adsorption: {str(ooh_reac)}' + '   %{y:.3f}',
                        marker=dict(color=colour_dict_metal[metal] if metal in colour_dict_metal.keys() else 'Grey',
                                    opacity=0.5, ),
                        legendgroup=metal,
                        legendgrouptitle_text=metal,
                    ))

                    fig.data = fig.data[-1:] + fig.data[0:-1]
                except:
                    traceback.print_exc()

    #if len(fig.data) > 0:
        #min_value = min([min(fig.data, key=lambda d: d['x'])['x'], min(fig.data, key=lambda d: d['y'])['y']])[0]
        #max_value = min([max(fig.data, key=lambda d: d['x'])['x'], max(fig.data, key=lambda d: d['y'])['y']])[0]

        #fig.add_shape(type='line',
        #              xref='x', yref='y',
        #              x0=min_value, y0=min_value + 3.2,
        #              x1=max_value, y1=max_value + 3.2,
        #              line=dict(color='grey', width=3, dash='solid'),
        #              opacity=0.5,
        #              layer='below',
        #              visible=True
        #              )


    fig.update_layout(
        title='Scaling of OOH and OH',
        xaxis_title='OH adsorption energy in reference to Pt_{111} adsorption',
        yaxis_title='OOH adsorption energy in reference to Pt_{111} adsorption'
    )

    folder_exist('reaction_plots')
    save_name = 'reaction_plots/scaling_plot_pt-ref_simple'
    if png_bool: fig.write_image(save_name + '.png')
    fig.write_html(save_name + '.html', include_mathjax='cdn')


def main(slab_db_dir: list[str], adsorbate_db_dir: list[str], mol_db_dir: list[str], thermo_dynamics: bool = False):
    pd_adsorbate_dat = build_pd(adsorbate_db_dir)
    pd_slab_dat = build_pd(slab_db_dir)
    pd_mol_dat = build_pd(mol_db_dir)

    functional_set = {xc for _, row in pd_adsorbate_dat.iterrows() if not pd.isna((xc := row.get('xc')))}

    oh_ad_h2_water = metal_ref_ractions[0::2] #adsorption_OH_reactions[1::3] #[1,4,7,10,13,16]
    ooh_ad_h2_water = metal_ref_ractions[1::2] #adsorption_OOH_reactions[1::3]

    dictionary_of_needed_strucs = {'molecule': [], 'slab': [], 'adsorbate': []}
    for reac in oh_ad_h2_water + ooh_ad_h2_water:
        for compo in reac.reactants + reac.products:
            dictionary_of_needed_strucs[compo.type].append(compo.name)

#    beef_vdw = Functional(functional_name='BEED-vdW', slab_db=pd_slab_dat, adsorbate_db=pd_adsorbate_dat, mol_db=pd_mol_dat, needed_struc_dict=dictionary_of_needed_strucs, thermo_dynamic=thermo_dynamics)

    functional_list = []
    for xc in functional_set:
        try: functional_list.append(Functional(functional_name=xc, slab_db=pd_slab_dat, adsorbate_db=pd_adsorbate_dat, mol_db=pd_mol_dat, needed_struc_dict=dictionary_of_needed_strucs, thermo_dynamic=thermo_dynamics))
        except: pass

    scaling_plot(functional_list, oh_ad_h2_water, ooh_ad_h2_water)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-adb', '--adsorbate_db', nargs='+', help='path to one or more databases containing the data.')
    parser.add_argument('-mdb', '--molecule_db', nargs='+', help='path to one or more databases containing the data.')
    parser.add_argument('-sdb', '--slab_db', nargs='+', help='path to one or more databases containing the data.')
    args = parser.parse_args()

    main(slab_db_dir=args.slab_db, adsorbate_db_dir=args.adsorbate_db, mol_db_dir=args.molecule_db)

