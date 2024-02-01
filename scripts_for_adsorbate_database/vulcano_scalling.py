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


def scaling_fits(O_ads_e: Sequence[float], OH_ads_e: Sequence[float], OOH_ads_e: Sequence[float], O_ads_e_sigma: Optional[Sequence[float]] = None, OH_ads_e_sigma: Optional[Sequence[float]] = None, OOH_ads_e_sigma: Optional[Sequence[float]] = None):
    if any(sigma_val is not None for sigma_val in (O_ads_e_sigma, OH_ads_e_sigma, OOH_ads_e_sigma)):
        @dataclass
        class odr_linear_fitting_results:
            slope: float
            stderr: float
            intercept: float
            intercept_stderr: float

        fitting_model = odr.Model(lambda beta, x: Linear_func(x, a=beta[0], b=beta[1]))
        OH_O_odr_dat = odr.Data(x=OH_ads_e, wd=1/(np.power(OH_ads_e_sigma, 2)), y=O_ads_e, we=1/(np.power(O_ads_e_sigma, 2))) #stats.linregress(x=OH_ads_e, y=O_ads_e)
        OH_O_odr_fit = odr.ODR(OH_O_odr_dat, fitting_model, beta0=[2, 0.5]).run() # beta0 is the initial guess for the scalling relation
        OH_O_fit_result = odr_linear_fitting_results(slope=OH_O_odr_fit.beta[0], intercept=OH_O_odr_fit.beta[1], stderr=OH_O_odr_fit.sd_beta[0], intercept_stderr=OH_O_odr_fit.sd_beta[1])

        OH_OOH_odr_dat = odr.Data(x=OH_ads_e, wd=1 / (np.power(OH_ads_e_sigma, 2)), y=OOH_ads_e, we=1 / (np.power(OOH_ads_e_sigma, 2)))
        OH_OOH_odr_fit = odr.ODR(OH_OOH_odr_dat, fitting_model, beta0=[2, 0.5]).run()  # beta0 is the initial guess for the scalling relation
        OH_OOH_fit_result = odr_linear_fitting_results(slope=OH_OOH_odr_fit.beta[0], intercept=OH_OOH_odr_fit.beta[1], stderr=OH_OOH_odr_fit.sd_beta[0], intercept_stderr=OH_OOH_odr_fit.sd_beta[1])
    else:
        OH_O_fit_result = stats.linregress(x=OH_ads_e, y=O_ads_e)
        OH_OOH_fit_result = stats.linregress(x=OH_ads_e, y=OOH_ads_e)
    return OH_O_fit_result, OH_OOH_fit_result


def Linear_func(x: float, a: float, b: float) -> float: return a*x+b


def Linear_func_err_square(x: float, x_sigma: float, a: float, a_sigma: float, b: float, b_sigma: float) -> float:
    return a**2 * x_sigma**2 + x**2 * a_sigma**2 + b_sigma**2


def overpotential_err_square(dG_OOH: float, dG_OOH_sigma: float, dG_OH: float, dG_OH_sigma: float, dG_O: float, dG_O_sigma: float) -> float:
    return min((
        (dG_OOH_sigma**2, 4.92 - dG_OOH), (dG_OOH_sigma**2 + dG_O_sigma**2, dG_OOH - dG_O), (dG_O_sigma**2 + dG_OH_sigma**2, dG_O - dG_OH), (dG_OH_sigma**2, dG_OH)
    ), key=itemgetter(1))[0]


def scaling_vulcano(functional_list: Sequence[Functional], o_reactions: Sequence[adsorbate_reaction], oh_reactions: Sequence[adsorbate_reaction], ooh_reactions: Sequence[adsorbate_reaction], png_bool: bool = False):
    fig = go.Figure()

    colour_dict_functional = {
        'PBE': '#CD5C5C',
        'RPBE': '#B22222',
        'PBE-PZ-SIC': '#FF8C00',
        'BEEF-vdW': '#0000CD',
        "{'name':'BEEF-vdW','backend':'libvdwxc'}": '#9370DB',
        'TPSS': px.colors.qualitative.Dark2[5]
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

    line = np.linspace(0, 2, 500)

    for xc in functional_list:
        line_arg = dict(line=dict(color=colour_dict_functional[xc.name],)) if xc.name in colour_dict_functional.keys() else dict(line=dict(color='DarkSlateGrey'))
        try:
            oh_o_fit, oh_ooh_fit = scaling_fits(
                O_ads_e=(o_adsor := list(map(xc.calculate_reaction_enthalpy, o_reactions))),
                OH_ads_e=(oh_adsor := list(map(xc.calculate_reaction_enthalpy, oh_reactions))),
                OOH_ads_e=(ooh_adsor := list(map(xc.calculate_reaction_enthalpy, ooh_reactions))),
                O_ads_e_sigma=[sd(ens) for ens in map(xc.calculate_BEE_reaction_enthalpy, o_reactions)] if xc.has_BEE else None,
                OH_ads_e_sigma=[sd(ens) for ens in map(xc.calculate_BEE_reaction_enthalpy, oh_reactions)] if xc.has_BEE else None,
                OOH_ads_e_sigma=[sd(ens) for ens in map(xc.calculate_BEE_reaction_enthalpy, ooh_reactions)] if xc.has_BEE else None

            )

            fig.add_trace(go.Scatter(mode='lines+markers',
                                     x=list(line + 0.35 - 0.5),
                                     y=list(map(lambda o, oh, ooh: overpotential(dG_O=o+ 0.05, #0.05 is dZPE - TdS from 10.1021/acssuschemeng.8b04173
                                                                                 dG_OH=oh + 0.35 - 0.5, #+ 0.35 is dZPE - TdS from 10.1021/jp047349j, - 0.3 is water stability correction 10.1021/cs300227s
                                                                                 dG_OOH=ooh + 0.40 - 0.3),  # same source as OH
                                                map(lambda x: Linear_func(x, oh_o_fit.slope, oh_o_fit.intercept), list(line)),
                                                list(line),
                                                map(lambda x: Linear_func(x, oh_ooh_fit.slope, oh_ooh_fit.intercept), list(line)))),
                                     name='linier scalling fit of ' + xc.name,
                                     hovertemplate=f'XC: {xc.name}',
                                     **line_arg,
                                     marker=dict(opacity=0)
                                     ))

            fig.update_traces(selector=dict(name='linier scalling fit of ' + xc.name),
                                      error_y=dict(type='data',
                                                   array=list(map(lambda o,o_sigma, oh, oh_sigma, ooh, ooh_sigma: np.sqrt(overpotential_err_square(
                                                       dG_O=o+0.05, #0.05 is dZPE - TdS from 10.1021/acssuschemeng.8b04173
                                                       dG_OH=oh + 0.35 - 0.5, #+ 0.35 is dZPE - TdS from 10.1021/jp047349j, - 0.3 is water stability correction 10.1021/cs300227s
                                                       dG_OOH=ooh + 0.40 - 0.3,# same source as OH
                                                       dG_O_sigma=np.sqrt(o_sigma),
                                                       dG_OH_sigma=np.sqrt(oh_sigma),
                                                       dG_OOH_sigma=np.sqrt(ooh_sigma))),
                                                                  map(lambda x: Linear_func(x, oh_o_fit.slope,
                                                                                            oh_o_fit.intercept), list(line)),  # the O fit
                                                                  map(lambda x: Linear_func_err_square(x, 0,
                                                                                                       oh_o_fit.slope,
                                                                                                       oh_o_fit.stderr,
                                                                                                       oh_o_fit.intercept,
                                                                                                       oh_o_fit.intercept_stderr), line),
                                                                  list(line),
                                                                  [0] * len(line),  # assuming that all the error is on the O and OOH relative to OH
                                                                  map(lambda x: Linear_func(x, oh_ooh_fit.slope,
                                                                                            oh_ooh_fit.intercept), list(line)),
                                                                  map(lambda x: Linear_func_err_square(x, 0,
                                                                                                       oh_ooh_fit.slope,
                                                                                                       oh_ooh_fit.stderr,
                                                                                                       oh_ooh_fit.intercept,
                                                                                                       oh_ooh_fit.intercept_stderr), line))),
                                                   color=colour_dict_functional[xc.name] if xc.name in colour_dict_functional.keys() else 'DarkSlateGrey', thickness=1.5, width=3, visible=True),
                                      )

        except: traceback.print_exc()

        #if xc.has_BEE:
        #    try:
        #        o_ensemble = map(xc.calculate_BEE_reaction_enthalpy, o_reactions)
        #        oh_ensemble = map(xc.calculate_BEE_reaction_enthalpy, oh_reactions)  # is a nested matrix like object, with rows corresponding the metals and col as each ensamble function
        #        ooh_ensemble = map(xc.calculate_BEE_reaction_enthalpy, ooh_reactions)

        #        ensemble_fits_oh_o, ensemble_fits_oh_ooh = tuple(zip(map(scaling_fits, o_ensemble, oh_ensemble, ooh_ensemble)))

        #        for scalling in (ensemble_fits_oh_o, ensemble_fits_oh_ooh):

        #            ens_slope_bins = np.histogram(map(attrgetter('slope'), scalling), bins='auto')
        #            ens_intercept_bins = np.histogram(map(attrgetter('intercept'), scalling), bins='auto')

        #    except: traceback.print_exc()

    for oh_reac, ooh_reac, o_reac in zip(oh_reactions, ooh_reactions, o_reactions):
        assert (metal := oh_reac.products[0].name.split('_')[0]) == ooh_reac.products[0].name.split('_')[0]
        for xc in functional_list:
            marker_arg = dict(marker=dict(color=colour_dict_metal[metal], size=16, line=dict(width=2,color=colour_dict_functional[xc.name] if xc.name in colour_dict_functional.keys() else 'DarkSlateGrey'))) if metal in colour_dict_metal.keys() else dict(marker=dict(size=16, line=dict(width=2, color='DarkSlateGrey')))
            try: fig.add_trace(go.Scatter(
                mode='markers',
                name=f'{xc.name}-{metal}',
                x=[(oh_adsorp := xc.calculate_reaction_enthalpy(oh_reac)) + 0.35 - 0.5], # + 0.35 is dZPE - TdS from 10.1021/jp047349j, - 0.3 is water stability correction 10.1021/cs300227s
                y=[overpotential(
                    dG_OOH=(ooh_adsorp := xc.calculate_reaction_enthalpy(ooh_reac)) + 0.40 - 0.3,
                    dG_OH=oh_adsorp + 0.35 - 0.5,
                    dG_O=xc.calculate_reaction_enthalpy(o_reac) + 0.05# oh_adsorp*2 + 0.05 # 0.05 is dZPE - TdS from 10.1021/acssuschemeng.8b04173
                )],
                hovertemplate=f'functional: {xc.name}' + '<br>' + f'metal: {metal}' + '<br>' + f'OH adsorption: {str(oh_reac)}' + '<br>' + f'OOH adsorption: {str(ooh_reac)}' + '<br>' + f'O adsorption: {str(o_reac)} ',
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
                        y=(ens_y_cloud := list(map(lambda ooh, oh, o: overpotential(
                            dG_OOH=ooh + 0.40 - 0.3,
                            dG_OH=oh + 0.35 - 0.5,
                            dG_O=o + 0.05
                        ),
                                                   xc.calculate_BEE_reaction_enthalpy(ooh_reac).tolist(),
                                                   (oh_ensem := xc.calculate_BEE_reaction_enthalpy(oh_reac)).tolist(),
                                                   xc.calculate_BEE_reaction_enthalpy(o_reac).tolist()
                                                   ))),
                        x=(ens_x_cloud := oh_ensem + 0.35 - 0.5),
                        hovertemplate=f'metal: {metal}' + '<br>' + f'OH adsorption: {str(oh_reac)}' + '<br>' + f'OOH adsorption: {str(ooh_reac)}',
                        marker=dict(color=colour_dict_metal[metal] if metal in colour_dict_metal.keys() else 'Grey', opacity=0.5, ),
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
                        args=[{"visible": [True if match(f'BEE for [A-Z][a-z] BEEF-vdW', trace.name) else 'undefined' for i, trace in enumerate(fig.data) if 'fit' not in trace.name],
                               'error_x.visible': [False if match('BEEF-vdW-[A-Z][a-z]', trace.name) else 'undefined' for i, trace in enumerate(fig.data) if 'fit' not in trace.name],
                               'error_y.visible': [False if match('BEEF-vdW-[A-Z][a-z]', trace.name) else 'undefined' for i, trace in enumerate(fig.data) if 'fit' not in trace.name],
                               }, [i for i, trace in enumerate(fig.data) if 'fit' not in trace.name]],
                        label='Ensemble',
                        method='restyle',
                    ),
                    dict(
                        args=[{"visible": [False if match(f'BEE for [A-Z][a-z] BEEF-vdW', trace.name) else 'undefined' for i, trace in enumerate(fig.data) if 'fit' not in trace.name],
                               'error_x.visible': [True if match('BEEF-vdW-[A-Z][a-z]', trace.name) else 'undefined' for i, trace in enumerate(fig.data) if 'fit' not in trace.name],
                               'error_y.visible': [True if match('BEEF-vdW-[A-Z][a-z]', trace.name) else 'undefined' for i, trace in enumerate(fig.data) if 'fit' not in trace.name],
                               }, [i for i, trace in enumerate(fig.data) if 'fit' not in trace.name]],
                        label='Error bars',
                        method='restyle',
                    ),
                    dict(
                        args=[{"visible": [True if match(f'BEE for [A-Z][a-z] BEEF-vdW', trace.name) else 'undefined' for i, trace in enumerate(fig.data) if 'fit' not in trace.name],
                               'error_x.visible': [True if match('BEEF-vdW-[A-Z][a-z]', trace.name) else 'undefined' for i, trace in enumerate(fig.data) if 'fit' not in trace.name],
                               'error_y.visible': [True if match('BEEF-vdW-[A-Z][a-z]', trace.name) else 'undefined' for i, trace in enumerate(fig.data) if 'fit' not in trace.name],
                               }, [i for i, trace in enumerate(fig.data) if 'fit' not in trace.name]],
                        label='Both',
                        method='restyle',
                    ),
                    dict(
                        args=[{"visible": [False if match(f'BEE for [A-Z][a-z] BEEF-vdW', trace.name) else 'undefined' for i, trace in enumerate(fig.data) if 'fit' not in trace.name],
                               'error_x.visible': [False if match('BEEF-vdW-[A-Z][a-z]', trace.name) else 'undefined' for i, trace in enumerate(fig.data) if 'fit' not in trace.name],
                               'error_y.visible': [False if match('BEEF-vdW-[A-Z][a-z]', trace.name) else 'undefined' for i, trace in enumerate(fig.data) if 'fit' not in trace.name],
                               }, [i for i, trace in enumerate(fig.data) if 'fit' not in trace.name]],
                        label='None',
                        method='restyle',
                    ),
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.5,
                xanchor="left",
                y=1.065,
                yanchor="top"
            ),])

    folder_exist('reaction_plots')
    save_name = 'reaction_plots/vulcano_fitted_plot'
    if png_bool: fig.write_image(save_name + '.png')
    fig.write_html(save_name + '.html', include_mathjax='cdn')


def main(slab_db_dir: list[str], adsorbate_db_dir: list[str], mol_db_dir: list[str], thermo_dynamics: bool = False):
    pd_adsorbate_dat = build_pd(adsorbate_db_dir)
    pd_slab_dat = build_pd(slab_db_dir)
    pd_mol_dat = build_pd(mol_db_dir)

    functional_set = {xc for _, row in pd_adsorbate_dat.iterrows() if not pd.isna((xc := row.get('xc')))}

    o_ad_h2_water = adsorption_O_reactions[1::3]
    oh_ad_h2_water = adsorption_OH_reactions[1::3]  # metal_ref_ractions[0::2] #adsorption_OH_reactions[1::3] #[1,4,7,10,13,16]
    ooh_ad_h2_water = adsorption_OOH_reactions[1::3]  # metal_ref_ractions[1::2] #adsorption_OOH_reactions[1::3]

    dictionary_of_needed_strucs = {'molecule': [], 'slab': [], 'adsorbate': []}
    for reac in oh_ad_h2_water + ooh_ad_h2_water + o_ad_h2_water:
        for compo in reac.reactants + reac.products:
            dictionary_of_needed_strucs[compo.type].append(compo.name)

    functional_list = []
    for xc in functional_set:
        try: functional_list.append(Functional(functional_name=xc, slab_db=pd_slab_dat, adsorbate_db=pd_adsorbate_dat, mol_db=pd_mol_dat, needed_struc_dict=dictionary_of_needed_strucs, thermo_dynamic=thermo_dynamics))
        except: pass

    scaling_vulcano(functional_list, oh_reactions=oh_ad_h2_water, ooh_reactions=ooh_ad_h2_water, o_reactions=o_ad_h2_water)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-adb', '--adsorbate_db', nargs='+', help='path to one or more databases containing the data.')
    parser.add_argument('-mdb', '--molecule_db', nargs='+', help='path to one or more databases containing the data.')
    parser.add_argument('-sdb', '--slab_db', nargs='+', help='path to one or more databases containing the data.')
    args = parser.parse_args()

    main(slab_db_dir=args.slab_db, adsorbate_db_dir=args.adsorbate_db, mol_db_dir=args.molecule_db)
