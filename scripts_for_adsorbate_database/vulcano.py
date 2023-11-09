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


def overpotential(dG_OOH: float, dG_OH: float, dG_O: float) -> float: return 1.23 - min((4.92 - dG_OOH, dG_OOH - dG_O, dG_O - dG_OH, dG_OH)) # dG_OOH - dG_O, dG_O - dG_OH


def vulcano_plotly(functional_list: Sequence[Functional], oh_reactions: Sequence[adsorbate_reaction], ooh_reactions: Sequence[adsorbate_reaction], png_bool: bool = False):
    fig = go.Figure()

    colour_dict = {
        'PBE': 'indianred',
        'RPBE': 'firebrick',
        'PBE-PZ-SIC': 'darkorange',
        'BEEF-vdW': 'mediumblue',
        "{'name':'BEEF-vdW','backend':'libvdwxc'}": 'mediumpurple'
    }

    line = np.linspace(0,2,500)
    over_potential_line = list(map(lambda x: overpotential(x + 3.2, x, x * 2), line))

    fig.add_trace(go.Scatter(
        mode='lines',
        x=line,
        y=over_potential_line,
        line=dict(
            color='Grey',
            #opacity=0.5
            ),
        showlegend=False,
    ))

    for xc in functional_list:
        marker_arg = dict(marker={'color': colour_dict[xc.name], 'size': 16}) if xc.name in colour_dict.keys() else dict(marker={'size': 16})
        for oh_reac, ooh_reac in zip(oh_reactions, ooh_reactions):
            try: fig.add_trace(go.Scatter(
                mode='markers',
                name=f'{xc.name}',
                x=[(oh_adsorp := xc.calculate_reaction_enthalpy(oh_reac))],
                y=[overpotential(
                    dG_OOH=(ooh_adsorp := xc.calculate_reaction_enthalpy(ooh_reac)),
                    dG_OH=oh_adsorp,
                    dG_O=oh_adsorp*2
                )],
                #hoverinfo=f'functional: {xc.name}',
                **marker_arg
            ))
            except: traceback.print_exc()

            if xc.name in ['BEEF-vdW', "{'name':'BEEF-vdW','backend':'libvdwxc'}"]:
                try:
                    fig.add_trace(go.Scatter(
                    mode='markers',
                    name=f'BEE for {xc.name}',
                    y=list(map(lambda ooh, oh: overpotential(
                        dG_OOH=ooh,
                        dG_OH=oh,
                        dG_O=oh*2),
                        xc.calculate_BEE_reaction_enthalpy(ooh_reac).tolist(),
                        (oh_ensem := xc.calculate_BEE_reaction_enthalpy(oh_reac).tolist()),)),
                    x=oh_ensem,
                    marker=dict(color='Grey', opacity=0.5, )
                    ))

                    fig.data = fig.data[-1:] + fig.data[0:-1]
                except: pass

    folder_exist('reaction_plots')
    save_name = 'reaction_plots/vulcano_plot'
    if png_bool: fig.write_image(save_name + '.png')
    fig.write_html(save_name + '.html', include_mathjax='cdn')


def main(slab_db_dir: list[str], adsorbate_db_dir: list[str], mol_db_dir: list[str], thermo_dynamics: bool = False):
    pd_adsorbate_dat = build_pd(adsorbate_db_dir)
    pd_slab_dat = build_pd(slab_db_dir)
    pd_mol_dat = build_pd(mol_db_dir)

    functional_set = {xc for _, row in pd_adsorbate_dat.iterrows() if not pd.isna((xc := row.get('xc')))}

    oh_ad_h2_water = adsorption_OH_reactions[1::3] #[1,4,7,10,13,16]
    ooh_ad_h2_water = adsorption_OH_reactions[1::3]

    dictionary_of_needed_strucs = {'molecule': [], 'slab': [], 'adsorbate': []}
    for reac in oh_ad_h2_water + ooh_ad_h2_water:
        for compo in reac.reactants + reac.products:
            dictionary_of_needed_strucs[compo.type].append(compo.name)

    functional_list = []
    for xc in functional_set:
        try: functional_list.append(Functional(functional_name=xc, slab_db=pd_slab_dat, adsorbate_db=pd_adsorbate_dat, mol_db=pd_mol_dat, needed_struc_dict=dictionary_of_needed_strucs, thermo_dynamic=thermo_dynamics))
        except: pass

    vulcano_plotly(functional_list, oh_ad_h2_water, ooh_ad_h2_water)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-adb', '--adsorbate_db', nargs='+', help='path to one or more databases containing the data.')
    parser.add_argument('-mdb', '--molecule_db', nargs='+', help='path to one or more databases containing the data.')
    parser.add_argument('-sdb', '--slab_db', nargs='+', help='path to one or more databases containing the data.')
    args = parser.parse_args()

    main(slab_db_dir=args.slab_db, adsorbate_db_dir=args.adsorbate_db, mol_db_dir=args.molecule_db)
