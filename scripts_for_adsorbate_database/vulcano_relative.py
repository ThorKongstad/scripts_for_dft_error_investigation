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


def overpotential(dG_OOH: float, dG_OH: float, dG_O: float) -> float: return min((4.92 - dG_OOH, dG_OOH - dG_O, dG_O - dG_OH, dG_OH)) # 1.23 -


def vulcano_plotly(functional_list: Sequence[Functional], oh_reactions: Sequence[adsorbate_reaction], ooh_reactions: Sequence[adsorbate_reaction], oh_reaction_relative: adsorbate_reaction, ooh_reactions_relative: Sequence[adsorbate_reaction],  png_bool: bool = False):
    fig = go.Figure()

    colour_dict_functional = {
        'PBE': 'indianred',
        'RPBE': 'firebrick',
        'PBE-PZ-SIC': 'darkorange',
        'BEEF-vdW': 'mediumblue',
        "{'name':'BEEF-vdW','backend':'libvdwxc'}": 'mediumpurple'
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

    #line = np.linspace(0, 2, 500) # used for dG_*OH = E_*OH - 0.05
    #over_potential_line = list(map(lambda x: overpotential(dG_OOH=(x - 0.05) + 3.2 + 0.40 - 0.3, dG_OH=x, dG_O=(x - 0.05) * 2) + 0.05, line))

#    fig.add_trace(go.Scatter(
#        mode='lines',
#        x=line,
#        y=over_potential_line,
#        line=dict(
#            color='Grey',
#           #opacity=0.5
#      ),
#      showlegend=False,
#    ))

    for oh_reac, ooh_reac in zip(oh_reactions, ooh_reactions):
        assert (metal := oh_reac.products[0].name.split('_')[0]) == ooh_reac.products[0].name.split('_')[0]
        marker_arg = dict(marker=dict(color=colour_dict_metal[metal], size=16, line=dict(width=2, color='DarkSlateGrey'))) if metal in colour_dict_metal.keys() else dict(marker=dict(size=16, line=dict(width=2, color='DarkSlateGrey')))
        for xc in functional_list:
            #marker_arg = dict(marker={'color': colour_dict[xc.name], 'size': 16}) if xc.name in colour_dict.keys() else dict(marker={'size': 16})
            try: fig.add_trace(go.Scatter(
                mode='markers',
                name=f'{xc.name}-{metal}',
                x=[(oh_adsorp := xc.calculate_reaction_enthalpy(oh_reac)) + 0.35 - 0.3 - ((oh_adsorp_relative := xc.calculate_reaction_enthalpy(oh_reaction_relative)) + 0.35 - 0.3)], # + 0.35 is dZPE - TdS from 10.1021/jp047349j, - 0.3 is water stability correction 10.1021/acssuschemeng.8b04173
                y=[overpotential(
                    dG_OOH=(ooh_adsorp := xc.calculate_reaction_enthalpy(ooh_reac)) + 0.40 - 0.3,
                    dG_OH=oh_adsorp + 0.35 - 0.3,
                    dG_O=oh_adsorp*2 + 0.05 # 0.05 is dZPE - TdS from 10.1021/acssuschemeng.8b04173
                )
                - overpotential(
                    dG_OOH=(xc.calculate_reaction_enthalpy(ooh_reactions_relative)) + 0.40 - 0.3,
                    dG_OH=oh_adsorp_relative + 0.35 - 0.3,
                    dG_O=oh_adsorp_relative *2 + 0.05 # 0.05 is dZPE - TdS from 10.1021/acssuschemeng.8b04173
                )],
                hovertemplate=f'functional: {xc.name}' + '<br>' + f'metal: {metal}' + '<br>' + f'OH adsorption: {str(oh_reac)}' + '<br>' + f'OOH adsorption: {str(ooh_reac)}',
                legendgroup=metal,
                legendgrouptitle_text=metal,
                **marker_arg
            ))
            except: traceback.print_exc()

            if xc.name in ['BEEF-vdW', "{'name':'BEEF-vdW','backend':'libvdwxc'}"]:
                try:
                    fig.add_trace(go.Scatter(
                        mode='markers',
                        name=f'BEE for {metal} {xc.name}',
                        y=np.array(list(map(lambda ooh, oh: overpotential(
                                dG_OOH=ooh + 0.40 - 0.3,
                                dG_OH=oh + 0.35 - 0.3,
                                dG_O=oh*2 + 0.05
                                ),
                            xc.calculate_BEE_reaction_enthalpy(ooh_reac).tolist(),
                            (oh_ensem := xc.calculate_BEE_reaction_enthalpy(oh_reac)).tolist(),
                            )))
                          - np.array(list(map(lambda ooh, oh: overpotential(
                                dG_OOH=ooh + 0.40 - 0.3,
                                dG_OH=oh + 0.35 - 0.3,
                                dG_O=oh*2 + 0.05
                                ),
                            xc.calculate_BEE_reaction_enthalpy(ooh_reactions_relative).tolist(),
                            (oh_ensem_relative := xc.calculate_BEE_reaction_enthalpy(oh_reaction_relative)).tolist(),
                            ))),
                        x=oh_ensem + 0.35 - 0.3 - (oh_ensem_relative + 0.35 - 0.3),
                        hovertemplate=f'metal: {metal}' + '<br>' + f'OH adsorption: {str(oh_reac)}' + '<br>' + f'OOH adsorption: {str(ooh_reac)}',
                        marker=dict(color=colour_dict_metal[metal] if metal in colour_dict_metal.keys() else 'Grey', opacity=0.5, ),
                        legendgroup=metal,
                        legendgrouptitle_text=metal,
                    ))

                    fig.data = fig.data[-1:] + fig.data[0:-1]
                except: pass

    fig.update_layout(
        title='ORR',
        xaxis_title='$\Delta G_{*OH} - \Delta G_{Pt111*OH}$',# in reference to Pt_{111} adsorption',
        yaxis_title='Overpotential relative to Pt'
    )

    folder_exist('reaction_plots')
    #save_name = 'reaction_plots/vulcano_pt_ref_plot'
    save_name = 'reaction_plots/vulcano_plot_relative'
    if png_bool: fig.write_image(save_name + '.png')
    fig.write_html(save_name + '.html', include_mathjax='cdn')


def main(slab_db_dir: list[str], adsorbate_db_dir: list[str], mol_db_dir: list[str], thermo_dynamics: bool = False):
    pd_adsorbate_dat = build_pd(adsorbate_db_dir)
    pd_slab_dat = build_pd(slab_db_dir)
    pd_mol_dat = build_pd(mol_db_dir)

    functional_set = {xc for _, row in pd_adsorbate_dat.iterrows() if not pd.isna((xc := row.get('xc')))}

    #oh_ad_h2_water = metal_ref_ractions[0::2] #adsorption_OH_reactions[1::3] #[1,4,7,10,13,16]
    oh_ad_h2_water = adsorption_OH_reactions[1::3] #[1,4,7,10,13,16]

    #ooh_ad_h2_water = metal_ref_ractions[1::2] #adsorption_OOH_reactions[1::3]
    ooh_ad_h2_water = adsorption_OOH_reactions[1::3]

    dictionary_of_needed_strucs = {'molecule': [], 'slab': [], 'adsorbate': []}
    for reac in oh_ad_h2_water + ooh_ad_h2_water:
        for compo in reac.reactants + reac.products:
            dictionary_of_needed_strucs[compo.type].append(compo.name)

    functional_list = []
    for xc in functional_set:
        try: functional_list.append(Functional(functional_name=xc, slab_db=pd_slab_dat, adsorbate_db=pd_adsorbate_dat, mol_db=pd_mol_dat, needed_struc_dict=dictionary_of_needed_strucs, thermo_dynamic=thermo_dynamics))
        except: pass

    vulcano_plotly(functional_list, oh_ad_h2_water, ooh_ad_h2_water, oh_ad_h2_water[0], ooh_ad_h2_water[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-adb', '--adsorbate_db', nargs='+', help='path to one or more databases containing the data.')
    parser.add_argument('-mdb', '--molecule_db', nargs='+', help='path to one or more databases containing the data.')
    parser.add_argument('-sdb', '--slab_db', nargs='+', help='path to one or more databases containing the data.')
    args = parser.parse_args()

    main(slab_db_dir=args.slab_db, adsorbate_db_dir=args.adsorbate_db, mol_db_dir=args.molecule_db)
