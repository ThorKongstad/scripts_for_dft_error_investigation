import argparse
import math
import sys
import pathlib
from typing import Sequence, Optional
import traceback
from re import match

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from scripts_for_adsorbate_database import sanitize, folder_exist, build_pd, adsorbate_reaction, adsorption_OH_reactions, adsorption_OOH_reactions, metal_ref_ractions, adsorption_O_reactions, sd
from scripts_for_adsorbate_database.adsorbate_correlation_plot import Functional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def overpotential(dG_OOH: float, dG_OH: float, dG_O: float) -> float: return min((4.92 - dG_OOH, dG_OOH - dG_O, dG_O - dG_OH, dG_OH)) # 1.23 -


def vulcano_plotly(functional_list: Sequence[Functional], oh_reactions: Sequence[adsorbate_reaction], ooh_reactions: Sequence[adsorbate_reaction], png_bool: bool = False, pt_rel_bool: bool = False):
    fig = go.Figure()

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

    line = np.linspace(0, 2, 500) # used for dG_*OH = E_*OH - 0.05
    over_potential_line = list(map(lambda x: overpotential(dG_OOH= x + 3.2, dG_OH=x, dG_O=x * 2) , line))
    if not pt_rel_bool:
        fig.add_trace(go.Scatter(
            mode='lines',
            x=line,
            y=over_potential_line,
            line=dict(
                color='Grey',
                #opacity=0.5
            ),
            showlegend=False,
            name='expected scaling relation',
            hovertemplate='OOH = OH + 3.2 <br> O = OH*2'
        ))

    OH_corr = 0.35 - 0.5  # + 0.35 is dZPE - TdS from 10.1021/jp047349j, - 0.5 is water stability correction 10.1021/cs300227s
    OOH_corr = 0.4 - 0.3

    Pt_OH_reac = [oh_reac for oh_reac in oh_reactions if oh_reac.products[0].name.split('_')[0] == 'Pt'][0]
    Pt_OOH_reac = [ooh_reac for ooh_reac in ooh_reactions if ooh_reac.products[0].name.split('_')[0] == 'Pt'][0]

    for oh_reac, ooh_reac in zip(oh_reactions, ooh_reactions):
        assert (metal := oh_reac.products[0].name.split('_')[0]) == ooh_reac.products[0].name.split('_')[0]
        #marker_arg = dict(marker=dict(color=colour_dict_metal[metal], size=16, line=dict(width=2, color='DarkSlateGrey'))) if metal in colour_dict_metal.keys() else dict(marker=dict(size=16, line=dict(width=2, color='DarkSlateGrey')))
        for xc in functional_list:
            marker_arg = dict(marker=dict(size=16, color=colour_dict_metal[metal] if metal in colour_dict_metal.keys() else 'DarkSlateGrey', symbol=marker_dict_functional[xc.name] if xc.name in marker_dict_functional.keys() else 'circle'))
            try:
                fig.add_trace(go.Scatter(
                mode='markers',
                name=f'{xc.name}-{metal}',
                x=[(oh_adsorp := xc.calculate_reaction_enthalpy(oh_reac)) + OH_corr - ((pt_oh_adsorp := xc.calculate_reaction_enthalpy(Pt_OH_reac)) + OH_corr if pt_rel_bool else 0)], # + 0.35 is dZPE - TdS from 10.1021/jp047349j, - 0.3 is water stability correction 10.1021/cs300227s
                y=[overpotential(
                    dG_OOH=(ooh_adsorp := xc.calculate_reaction_enthalpy(ooh_reac)) + OOH_corr,
                    dG_OH=oh_adsorp + OH_corr,
                    dG_O=2*(oh_adsorp + OH_corr) # + 0.05# oh_adsorp*2 + 0.05 # 0.05 is dZPE - TdS from 10.1021/acssuschemeng.8b04173
                    )
                    -(overpotential(
                    dG_OOH=(pt_ooh_adsorp := xc.calculate_reaction_enthalpy(Pt_OOH_reac)) + OOH_corr,
                    dG_OH=pt_oh_adsorp + OH_corr,
                    dG_O=2*(pt_oh_adsorp + OH_corr)# + 0.05# oh_adsorp*2 + 0.05 # 0.05 is dZPE - TdS from 10.1021/acssuschemeng.8b04173
                    ) if pt_rel_bool else 0)],
                hovertemplate=f'functional: {xc.name}' + '<br>' + f'metal: {metal}' + '<br>' + f'OH adsorption: {str(oh_reac)}' + '<br>' + f'OOH adsorption: {str(ooh_reac)}' + '<br>' + f'O adsorption: 2*E_OH ' + '<br> G_OH: %{x:.3f} eV' + '<br> Limiting Potential: %{y:.3f} eV',
                legendgroup=metal,
                legendgrouptitle_text=metal,
                **marker_arg
                ))

                if metal == 'Pt' and (not pt_rel_bool or xc.name == 'BEEF-vdW'):
                    fig.add_vline(
                        x=(0 if pt_rel_bool else oh_adsorp) + 0.11,
                        line_dash='dash',
                        annotation_text=f'Expected volcano peak' + (f'for {xc.name}' if not pt_rel_bool else '')
                    )

            except: traceback.print_exc()

            if xc.has_BEE:
                try:
                    fig.add_trace(go.Scatter(
                        mode='markers',
                        name=f'BEE for {metal} {xc.name}',
                        y=(ens_y_cloud := np.array(list(map(lambda ooh, oh: overpotential(
                                dG_OOH=ooh + OOH_corr,
                                dG_OH=oh + OH_corr,
                                dG_O=2*(oh + OH_corr)
                                ),
                            xc.calculate_BEE_reaction_enthalpy(ooh_reac).tolist(),
                            (oh_ensem := xc.calculate_BEE_reaction_enthalpy(oh_reac)).tolist(),
                            #xc.calculate_BEE_reaction_enthalpy(o_reac).tolist()
                            )))
                                          - (np.array(list(map(lambda ooh, oh: overpotential(
                                dG_OOH=ooh + OOH_corr,
                                dG_OH=oh + OH_corr,
                                dG_O=2*(oh + OH_corr)
                                ),
                            xc.calculate_BEE_reaction_enthalpy(Pt_OOH_reac).tolist(),
                            (Pt_oh_ensem := xc.calculate_BEE_reaction_enthalpy(Pt_OH_reac)) .tolist()))) if pt_rel_bool else 0)),
                        x=(ens_x_cloud := oh_ensem + OH_corr - (Pt_oh_ensem + OH_corr if pt_rel_bool else 0)),
                        hovertemplate=f'metal: {metal}' + '<br>' + f'OH adsorption: {str(oh_reac)}' + '<br>' + f'OOH adsorption: {str(ooh_reac)}',
                        marker=dict(color=colour_dict_metal[metal] if metal in colour_dict_metal.keys() else 'Grey', opacity=0.5, ),
                        legendgroup=metal,
                        legendgrouptitle_text=metal,
                        visible=True
                    ))
                    fig.update_traces(selector=dict(name=f'{xc.name}-{metal}'),
                                      error_x_type='constant',
                                      error_y_type='constant',
                                      error_x_value=(x_err := sd(ens_x_cloud)),
                                      error_y_value=(y_err := sd(ens_y_cloud)),
                                      error_x_color=colour_dict_metal[metal] if metal in colour_dict_metal.keys() else 'Grey',
                                      error_y_color=colour_dict_metal[metal] if metal in colour_dict_metal.keys() else 'Grey',
                                      error_x_thickness=1.5,
                                      error_y_thickness=1.5,
                                      error_x_width=3,
                                      error_y_width=3,
                                      error_x_visible=False,
                                      error_y_visible=False,
                                      hovertemplate=f'functional: {xc.name}' + '<br>' + f'metal: {metal}' + '<br>' + f'OH adsorption: {str(oh_reac)}' + '<br>' + f'OOH adsorption: {str(ooh_reac)}' + '<br>' + f'O adsorption: 2*E_OH ' + '<br> G_OH: %{x:.3f}  +- ' + f'{x_err:.3f} eV' + '<br> Limiting Potential: %{y:.3f}  +- ' + f'{y_err:.3f} eV',
                                      #error_x=dict(type='constant', value=sd(ens_x_cloud), color=colour_dict_metal[metal] if metal in colour_dict_metal.keys() else 'Grey', thickness=1.5, width=3, visible=True),
                                      #error_y=dict(type='constant', value=sd(ens_y_cloud), color=colour_dict_metal[metal] if metal in colour_dict_metal.keys() else 'Grey', thickness=1.5, width=3, visible=True)
                                      )
                    fig.data = fig.data[-1:] + fig.data[0:-1]
                except: traceback.print_exc()

    fig.update_layout(
        title='ORR',
        xaxis_title='$\Delta G_{*OH}$' if not pt_rel_bool else '$\Delta G_{*OH}-\Delta G_{Pt*OH}$',# in reference to Pt_{111} adsorption',
        yaxis_title='Limiting potential' + (' - Platinum\'s Limiting' if pt_rel_bool else ''),

        updatemenus = [
            dict(
                type='buttons',
                direction='left',
                buttons=[
                    dict(
                        args=[{"visible": [True] * len(fig.data),
                               'error_x.visible': [False] * len(fig.data),
                               'error_y.visible': [False] * len(fig.data)}
                              ],
                        label='Ensemble',
                        method='update',
                    ),
                    dict(
                        args=[{"visible": [False if match(f'BEE for [A-Z][a-z] BEEF-vdW', trace.name) else True for trace in fig.data],
                               'error_x.visible': [True if match('BEEF-vdW-[A-Z][a-z]', trace.name) else False for trace in fig.data],
                               'error_y.visible': [True if match('BEEF-vdW-[A-Z][a-z]', trace.name) else False for trace in fig.data]
                               }],
                        label='Error bars',
                        method='update',
                    ),
                    dict(
                        args=[{"visible": [True] * len(fig.data),
                               'error_x.visible': [True if match('BEEF-vdW-[A-Z][a-z]', trace.name) else False for trace in fig.data],
                               'error_y.visible': [True if match('BEEF-vdW-[A-Z][a-z]', trace.name) else False for trace in fig.data]
                               }],
                        label='Both',
                        method='update',
                    ),
                    dict(
                        args=[{"visible": [False if match(f'BEE for [A-Z][a-z] BEEF-vdW', trace.name) else True for trace in fig.data],
                               'error_x.visible': [False] * len(fig.data),
                               'error_y.visible': [False] * len(fig.data)
                               }],
                        label='None',
                        method='update',
                    ),
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.5,
                xanchor="left",
                y=1.065,
                yanchor="top"
            )
        ]
    )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1
    )

    folder_exist('reaction_plots')
    #save_name = 'reaction_plots/vulcano_pt_ref_plot'
    save_name = 'reaction_plots/vulcano_plot_OH' + ('_abs_Pt_rel' if pt_rel_bool else '')
    if png_bool: fig.write_image(save_name + '.png')
    fig.write_html(save_name + '.html', include_mathjax='cdn')


def main(slab_db_dir: list[str], adsorbate_db_dir: list[str], mol_db_dir: list[str], thermo_dynamics: bool = False, pt_rel_bool: bool = False):
    pd_adsorbate_dat = build_pd(adsorbate_db_dir)
    pd_slab_dat = build_pd(slab_db_dir)
    pd_mol_dat = build_pd(mol_db_dir)

    functional_set = {xc for _, row in pd_adsorbate_dat.iterrows() if not pd.isna((xc := row.get('xc')))}

    #oh_ad_h2_water = metal_ref_ractions[0::2] #adsorption_OH_reactions[1::3] #[1,4,7,10,13,16]
    oh_ad_h2_water = adsorption_OH_reactions[1::3] #[1,4,7,10,13,16]

    #ooh_ad_h2_water = metal_ref_ractions[1::2] #adsorption_OOH_reactions[1::3]
    ooh_ad_h2_water = adsorption_OOH_reactions[1::3]

#    o_ad_h2_water = adsorption_O_reactions[1::3]

    dictionary_of_needed_strucs = {'molecule': [], 'slab': [], 'adsorbate': []}
    for reac in oh_ad_h2_water + ooh_ad_h2_water:
        for compo in reac.reactants + reac.products:
            dictionary_of_needed_strucs[compo.type].append(compo.name)

    functional_list = []
    for xc in functional_set:
        try: functional_list.append(Functional(functional_name=xc, slab_db=pd_slab_dat, adsorbate_db=pd_adsorbate_dat, mol_db=pd_mol_dat, needed_struc_dict=dictionary_of_needed_strucs, thermo_dynamic=thermo_dynamics))
        except: pass

    vulcano_plotly(functional_list, oh_ad_h2_water, ooh_ad_h2_water, pt_rel_bool=pt_rel_bool)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-adb', '--adsorbate_db', nargs='+', help='path to one or more databases containing the data.')
    parser.add_argument('-mdb', '--molecule_db', nargs='+', help='path to one or more databases containing the data.')
    parser.add_argument('-sdb', '--slab_db', nargs='+', help='path to one or more databases containing the data.')
    parser.add_argument('-r', '--Pt_relative', action='store_true', default=False)
    args = parser.parse_args()

    main(slab_db_dir=args.slab_db, adsorbate_db_dir=args.adsorbate_db, mol_db_dir=args.molecule_db, pt_rel_bool=args.Pt_relative)
