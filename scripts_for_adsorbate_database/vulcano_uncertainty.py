import argparse
import math
import sys
import pathlib
from typing import Sequence, Optional
import traceback
from re import match
from operator import itemgetter, attrgetter
from dataclasses import dataclass, field

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from scripts_for_adsorbate_database import sanitize, folder_exist, build_pd, adsorbate_reaction, adsorption_OH_reactions, adsorption_OOH_reactions, adsorption_O_reactions, metal_ref_ractions, sd, mean, overpotential
from scripts_for_adsorbate_database.adsorbate_correlation_plot import Functional

import numpy as np
from scipy import stats, odr
from scipy.optimize import curve_fit
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def scaling_fit(reac_1_Energies: Sequence[float], reac_2_Energies: Sequence[float], reac_1_Energies_sigma: Optional[Sequence[float]] = None, reac_2_Energies_sigma: Optional[Sequence[float]] = None):
    if any(sigma_val is not None for sigma_val in (reac_1_Energies_sigma, reac_2_Energies_sigma)):
        @dataclass
        class odr_linear_fitting_results:
            slope: float
            stderr: float
            intercept: float
            intercept_stderr: float

        fitting_model = odr.Model(lambda beta, x: Linear_func(x, a=beta[0], b=beta[1]))

        fit_odr_dat = odr.Data(x=reac_1_Energies, wd=1 / (np.power(reac_1_Energies_sigma, 2)), y=reac_2_Energies, we=1 / (np.power(reac_2_Energies_sigma, 2)))
        odr_fit_obj = odr.ODR(fit_odr_dat, fitting_model, beta0=[1, 3.2]).run()  # beta0 is the initial guess for the scalling relation
        fit_result = odr_linear_fitting_results(slope=odr_fit_obj.beta[0], intercept=odr_fit_obj.beta[1], stderr=odr_fit_obj.sd_beta[0], intercept_stderr=odr_fit_obj.sd_beta[1])
    else:
        fit_result = stats.linregress(x=reac_1_Energies, y=reac_2_Energies)
    return fit_result


def Linear_func(x: float, a: float, b: float) -> float: return a*x+b


def limiting_potential_Pt_ref(dG_OOH: float, dG_OH: float) -> float: return dG_OH if dG_OH <= 1.1 else 4.92 - dG_OOH


def uncertainty_vulcano(functional_list: Sequence[Functional], oh_reactions_gas_ref: Sequence[adsorbate_reaction], ooh_reactions_gas_ref: Sequence[adsorbate_reaction], oh_reactions: Sequence[adsorbate_reaction], ooh_reactions: Sequence[adsorbate_reaction], png_bool: bool = False):
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

    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        name='Pt reference',
        hoveron=False,
    ))

    beef = [xc for xc in functional_list if xc.name == 'BEEF_vdW'][0]

    OH_OOH_scalling_fit = scaling_fit([beef.calculate_reaction_enthalpy(OH_reac) for OH_reac in oh_reactions_gas_ref],
                                      [beef.calculate_reaction_enthalpy(OOH_reac) for OOH_reac in ooh_reactions_gas_ref],
                                      [sd(beef.calculate_BEE_reaction_enthalpy(OH_reac)) for OH_reac in oh_reactions_gas_ref],
                                      [sd(beef.calculate_BEE_reaction_enthalpy(OOH_reac)) for OOH_reac in ooh_reactions_gas_ref])

    fig.add_vline(
        x=1.1,
        x0=1.1 - OH_OOH_scalling_fit.intercept_stderr/2, x1=1.1 + OH_OOH_scalling_fit.intercept_stderr/2,
        line_dash='dash',
        fillcolor="green",
        opacity=0.25,
        annotation_text="Uncertainty of Vulcano location", annotation_position="top left"
    )

    for oh_reac, ooh_reac in zip(oh_reactions, ooh_reactions):
        assert (metal := oh_reac.products[0].name.split('_')[0]) == ooh_reac.products[0].name.split('_')[0]
        for xc in functional_list:
            marker_arg = dict(marker=dict(size=16, color=colour_dict_metal[metal] if metal in colour_dict_metal.keys() else 'DarkSlateGrey', symbol=marker_dict_functional[xc.name] if xc.name in marker_dict_functional.keys() else 'circle'))

            try: fig.add_trace(go.Scatter(
                mode='markers',
                name=f'{xc.name}-{metal}',
                x=[(oh_adsorp := xc.calculate_reaction_enthalpy(oh_reac))],
                y=[limiting_potential_Pt_ref(dG_OH=oh_adsorp, dG_OOH=xc.calculate_reaction_enthalpy(ooh_reac))],
                hovertemplate=f'functional: {xc.name}' + '<br>' + f'metal: {metal}' + '<br>' + f'OH adsorption: {str(oh_reac)}' + '<br>' + f'OOH adsorption: {str(ooh_reac)}',
                legendgroup=metal,
                legendgrouptitle_text=metal,
                **marker_arg
            ))
            except: traceback.print_exc()

            if xc.has_BEE:
                try:
                    fig.add_trace(go.Scatter(
                        mode='markers',
                        name=f'BEE for {metal} {xc.name}',
                        y=(ens_y_cloud := list(map(
                            lambda ooh, oh: limiting_potential_Pt_ref(
                                dG_OOH=ooh,
                                dG_OH=oh,
                                ),
                            xc.calculate_BEE_reaction_enthalpy(ooh_reac).tolist(),
                            (oh_ensem := xc.calculate_BEE_reaction_enthalpy(oh_reac)).tolist(),
                            ))),
                        x=(ens_x_cloud := oh_ensem),
                        hovertemplate=f'metal: {metal}' + '<br>' + f'OH adsorption: {str(oh_reac)}' + '<br>' + f'OOH adsorption: {str(ooh_reac)}',
                        marker=dict(color=colour_dict_metal[metal] if metal in colour_dict_metal.keys() else 'Grey', opacity=0.5, symbol=marker_dict_functional[xc.name] if xc.name in marker_dict_functional.keys() else 'circle'),
                        legendgroup=metal,
                        legendgrouptitle_text=metal,
                        visible=True
                    ))
                    fig.update_traces(selector=dict(name=f'{xc.name}-{metal}'),
                                      error_x_type='constant',
                                      error_y_type='constant',
                                      error_x_value=sd(ens_x_cloud),
                                      error_y_value=sd(ens_y_cloud),
                                      error_x_color=colour_dict_metal[metal] if metal in colour_dict_metal.keys() else 'Grey',
                                      error_y_color=colour_dict_metal[metal] if metal in colour_dict_metal.keys() else 'Grey',
                                      error_x_thickness=1.5,
                                      error_y_thickness=1.5,
                                      error_x_width=3,
                                      error_y_width=3,
                                      error_x_visible=False,
                                      error_y_visible=False,
                                      #error_x=dict(type='constant', value=sd(ens_x_cloud), color=colour_dict_metal[metal] if metal in colour_dict_metal.keys() else 'Grey', thickness=1.5, width=3, visible=True),
                                      #error_y=dict(type='constant', value=sd(ens_y_cloud), color=colour_dict_metal[metal] if metal in colour_dict_metal.keys() else 'Grey', thickness=1.5, width=3, visible=True)
                                      )
                    fig.data = fig.data[-1:] + fig.data[0:-1]
                except: traceback.print_exc()

    fig.update_layout(
        title='ORR',
        xaxis_title='$\Delta G_{*OH}$',  # in reference to Pt_{111} adsorption',
        yaxis_title='Limiting potential',

        updatemenus=[
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

    folder_exist('reaction_plots')
    #save_name = 'reaction_plots/vulcano_pt_ref_plot'
    save_name = 'reaction_plots/vulcano_plot_pt_ref'
    if png_bool: fig.write_image(save_name + '.png')
    fig.write_html(save_name + '.html', include_mathjax='cdn')

def main(slab_db_dir: list[str], adsorbate_db_dir: list[str], mol_db_dir: list[str], thermo_dynamics: bool = False):
    pd_adsorbate_dat = build_pd(adsorbate_db_dir)
    pd_slab_dat = build_pd(slab_db_dir)
    pd_mol_dat = build_pd(mol_db_dir)

    functional_set = {xc for _, row in pd_adsorbate_dat.iterrows() if not pd.isna((xc := row.get('xc')))}

    #o_ad_h2_water = adsorption_O_reactions[1::3]
    oh_ad_h2_water = adsorption_OH_reactions[1::3]  # metal_ref_ractions[0::2] #adsorption_OH_reactions[1::3] #[1,4,7,10,13,16]
    ooh_ad_h2_water = adsorption_OOH_reactions[1::3]  # metal_ref_ractions[1::2] #adsorption_OOH_reactions[1::3]

    #OH_pt_ref = adsorption_OH_reactions[1]
    #OOH_pt_ref = adsorption_OOH_reactions[1]

    oh_ad_pt_ref = metal_ref_ractions[0::2]
    ooh_ad_pt_ref = metal_ref_ractions[1::2]

    dictionary_of_needed_strucs = {'molecule': set(), 'slab': set(), 'adsorbate': set()}
    for reac in oh_ad_h2_water + ooh_ad_h2_water + oh_ad_pt_ref + ooh_ad_pt_ref:
        for compo in reac.reactants + reac.products:
            dictionary_of_needed_strucs[compo.type].add(compo.name)

    functional_list = []
    for xc in functional_set:
        try: functional_list.append(Functional(functional_name=xc, slab_db=pd_slab_dat, adsorbate_db=pd_adsorbate_dat, mol_db=pd_mol_dat, needed_struc_dict=dictionary_of_needed_strucs, thermo_dynamic=thermo_dynamics))
        except: pass

    uncertainty_vulcano(functional_list, oh_ad_h2_water, ooh_ad_h2_water, oh_ad_pt_ref, ooh_ad_pt_ref)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-adb', '--adsorbate_db', nargs='+', help='path to one or more databases containing the data.')
    parser.add_argument('-mdb', '--molecule_db', nargs='+', help='path to one or more databases containing the data.')
    parser.add_argument('-sdb', '--slab_db', nargs='+', help='path to one or more databases containing the data.')
    args = parser.parse_args()

    main(slab_db_dir=args.slab_db, adsorbate_db_dir=args.adsorbate_db, mol_db_dir=args.molecule_db)
