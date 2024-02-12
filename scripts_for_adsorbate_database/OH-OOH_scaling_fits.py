import argparse
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


def pearson_corr_coef(reac_1_Energies: Sequence[float], reac_2_Energies: Sequence[float], reac_1_Energies_sigma: Sequence[float],  reac_2_Energies_sigma: Sequence[float]) -> float:
    return covariance(reac_1_Energies, reac_2_Energies, (r1_Wmean := weighted_mean(reac_1_Energies, reac_1_Energies_sigma)), (r2_Wmean := weighted_mean(reac_2_Energies, reac_2_Energies_sigma))) \
           / (weighted_stderr(reac_1_Energies, reac_1_Energies_sigma, r1_Wmean) * weighted_stderr(reac_2_Energies, reac_2_Energies_sigma, r2_Wmean))


def weighted_mean(x_arr: Iterable[float], x_sigma_arr: Iterable[float]) -> float: return sum(x/(sigma**2) for x, sigma in zip(x_arr, x_sigma_arr))


def weighted_stderr(x_arr: Iterable[float], x_sigma_arr: Iterable[float], x_mean: Optional[float] = None) -> float:
    if x_mean is None: x_mean = weighted_mean(x_arr, x_sigma_arr)
    return np.sqrt(sum(((x-x_mean)**2)/sigma**2 for x, sigma in zip(x_arr, x_sigma_arr))/sum(1/sigma**2 for x, sigma in zip(x_arr, x_sigma_arr)))


def covariance(x_arr: Sequence[float], y_arr: Sequence[float], x_mean: Optional[float] = None, y_mean: Optional[float] = None) -> float:
    if x_mean is None: x_mean = mean(x_arr)
    if y_mean is None: y_mean = mean(y_arr)
    assert len(x_arr) == len(y_arr)
    return 1/len(x_arr) * sum((x - x_mean)*(y - y_mean) for x, y in zip(x_arr, y_arr))


def ode_1par_linear(reac_1_Energies: Sequence[float], reac_2_Energies: Sequence[float], reac_1_Energies_sigma: Optional[Sequence[float]] = None, reac_2_Energies_sigma: Optional[Sequence[float]] = None) -> 'odr_linear_fitting_results':
    @dataclass
    class odr_linear_fitting_results:
        slope: float
        stderr: float
        intercept: float
        intercept_stderr: float

    fitting_model = odr.Model(lambda beta, x: x+beta[0])

    fit_odr_dat = odr.Data(x=reac_1_Energies, wd=(1 / (np.power(reac_1_Energies_sigma, 2))) if reac_1_Energies_sigma is not None else None,
                           y=reac_2_Energies, we=(1 / (np.power(reac_2_Energies_sigma, 2))) if reac_1_Energies_sigma is not None else None)
    odr_fit_obj = odr.ODR(fit_odr_dat, fitting_model, beta0=[3.2]).run()  # beta0 is the initial guess for the scalling relation
    fit_result = odr_linear_fitting_results(slope=1, intercept=odr_fit_obj.beta[0], stderr=0, intercept_stderr=odr_fit_obj.sd_beta[0])
    return fit_result


def ensemble_histogram(data: Sequence[float], name, x_axis_title, marker_dict):
    fig = go.Figure()

    #bins, count = np.histogram(data,)
    #bins = 0.5 * (bins[:-1] + bins[1:])

    fig.add_trace(go.Histogram(x=data, name=name, marker=marker_dict, hovertemplate=f'Mean: {mean(data)}<br>Stderr: {sd(data)}'))
    #fig.add_trace(go.Bar(x=bins, y=count, name=name, marker=marker_dict))

    #fig.update_layout(
    #    xaxis_title=x_axis_title,
    #    yaxis_title='Count',)

    #fig.add_annotation(text=f'Mean: {mean(data)}<br>Stderr: {sd(data)}', xref="paper", yref="paper", showarrow=False)

    return fig


def scaling_plot(functional_list: Sequence[Functional], oh_reactions: Sequence[adsorbate_reaction], ooh_reactions: Sequence[adsorbate_reaction], png_bool: bool = False):
    def liniar_func(x: float, a: float, b: float) -> float: return a*x+b
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

    line = np.linspace(0, 2, 500)


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

    OH_adsorption_values = []
    OOH_adsorption_values = []
    fit_all_obj = []


    for xc in functional_list:
        line_arg = dict(line=dict(color=colour_dict_functional[xc.name],)) if xc.name in colour_dict_functional.keys() else dict(line=dict(color='DarkSlateGrey'))
        try:
            if not xc.has_BEE:
                #fit_obj = stats.linregress(x=(oh_adsor := list(map(xc.calculate_reaction_enthalpy, oh_reactions))),
                #                           y=(ooh_adsor := list(map(xc.calculate_reaction_enthalpy, ooh_reactions))),)
                fit_obj = ode_1par_linear(reac_1_Energies=(oh_adsor := list(map(xc.calculate_reaction_enthalpy, oh_reactions))),
                                          reac_2_Energies=(ooh_adsor := list(map(xc.calculate_reaction_enthalpy, ooh_reactions))),
                                          )

                fig.add_trace(go.Scatter(mode='lines',
                                         x=list(line),
                                         y=list(map(lambda x: liniar_func(x, fit_obj.slope, fit_obj.intercept), line)),
                                         name='linier scalling fit of ' + xc.name,
                                         hovertemplate=f'XC: {xc.name}'+'<br>'+f'Slope: {fit_obj.slope:.3f} +- {fit_obj.stderr:.3f}'+'<br>'+f'Intercept: {fit_obj.intercept:.3f} +- {fit_obj.intercept_stderr:.3f}'+'<br>',
                                         **line_arg
                                         ))

                OH_adsorption_values.extend(oh_adsor)
                OOH_adsorption_values.extend(ooh_adsor)
                fit_all_obj.append(fit_obj)

        except: traceback.print_exc()

        if xc.has_BEE:
            try:
                oh_ensamble = list(map(xc.calculate_BEE_reaction_enthalpy, oh_reactions)) # is a nested matrix like object, with rows corresponding the metals and col as each ensamble function
                ooh_ensamble = list(map(xc.calculate_BEE_reaction_enthalpy, ooh_reactions))
                fit_ens_objs = [ode_1par_linear(reac_1_Energies=OH_vals, reac_2_Energies=OOH_vals) for OH_vals, OOH_vals in zip(zip(*oh_ensamble), zip(*ooh_ensamble))]#[stats.linregress(x=OH_vals, y=OOH_vals) for OH_vals, OOH_vals in zip(zip(*oh_ensamble), zip(*ooh_ensamble))]

                for i, fit in enumerate(fit_ens_objs):
                    fig.add_trace(go.Scatter(mode='lines',
                                             x=list(line),
                                             y=list(map(lambda x: liniar_func(x, fit.slope, fit.intercept), line)),
                                             name=f'BEE fits No. {i} for ' + xc.name,
                                             legendgroup='BEE fits for ' + xc.name,
                                             legendgrouptitle_text='BEE fits for ' + xc.name,
                                             hovertemplate=f'XC: BEE No. {i} for {xc.name}'+'<br>'+f'Slope: {fit.slope:.3f} +- {fit.stderr:.3f}'+'<br>'+f'Intercept: {fit.intercept:.3f} +- {fit.intercept_stderr:.3f}'+'<br>',
                                             line=dict(color=colour_dict_functional[xc.name] if xc.name in colour_dict_functional.keys() else 'Grey',),
                                             opacity=0.05,
                                             showlegend=False
                                             ))
                    fig.data = fig.data[-1:] + fig.data[0:-1]

                fit_all_obj.extend(fit_ens_objs)
                for oh_row, ooh_row in zip(oh_ensamble,ooh_ensamble):
                    OH_adsorption_values.extend(oh_row)
                    OOH_adsorption_values.extend(ooh_row)

                fit_obj = ode_1par_linear(
                    reac_1_Energies=(oh_adsor := list(map(xc.calculate_reaction_enthalpy, oh_reactions))),
                    reac_1_Energies_sigma=[sd(ens) for ens in oh_ensamble],
                    reac_2_Energies=(ooh_adsor := list(map(xc.calculate_reaction_enthalpy, ooh_reactions))),
                    reac_2_Energies_sigma=[sd(ens) for ens in ooh_ensamble],
                )

                fig.add_trace(go.Scatter(mode='lines',
                                         x=list(line),
                                         y=list(map(lambda x: liniar_func(x, fit_obj.slope, fit_obj.intercept), line)),
                                         name='linier scalling fit of ' + xc.name,
                                         hovertemplate=f'XC: {xc.name}'+'<br>'+f'Slope: {fit_obj.slope:.3f} +- {fit_obj.stderr:.3f}'+'<br>'+f'Intercept: {fit_obj.intercept:.3f} +- {fit_obj.intercept_stderr:.3f}'+'<br>',
                                         **line_arg
                                         ))
                OH_adsorption_values.extend(oh_adsor)
                OOH_adsorption_values.extend(ooh_adsor)
                fit_all_obj.append(fit_obj)

                ens_figure = ensemble_histogram(list(fit.intercept for fit in fit_ens_objs), f'{xc.name} bee histogram', f'intercept values: {xc.name} bee', marker_dict=dict(color=colour_dict_functional[xc.name] if xc.name in colour_dict_functional.keys() else 'DarkSlateGrey'))

            except: traceback.print_exc()

    #Concatenated_fit = stats.linregress(x=OH_adsorption_values, y=OOH_adsorption_values)
    #fig.add_trace(go.Scatter(
    #    mode='lines',
    #    x=list(line),
    #    y=list(map(lambda x: liniar_func(x, Concatenated_fit.slope, Concatenated_fit.intercept), line)),
    #    name=f'Concatenated fit of all data points',
    #    hovertemplate=f'Concatenated fit' + '<br>' + f'Slope: {Concatenated_fit.slope:.3f} +- {Concatenated_fit.stderr:.3f}' + '<br>' + f'Intercept: {Concatenated_fit.intercept:.3f} +- {Concatenated_fit.intercept_stderr:.3f}' + '<br>' + f'R-square: {Concatenated_fit.rvalue:.3f}',
    #    line=dict(color='Black',)
    #))

    #alpha_mean, alpha_stderr = sum(fit_i.slope/(fit_i.stderr**2) for fit_i in fit_all_obj)/sum(1/(fit_i.stderr**2) for fit_i in fit_all_obj), np.sqrt(1/sum(1/(fit_i.stderr**2) for fit_i in fit_all_obj))
    #beta_mean, beta_stderr = sum(fit_i.intercept/(fit_i.intercept_stderr**2) for fit_i in fit_all_obj)/sum(1/(fit_i.intercept_stderr**2) for fit_i in fit_all_obj), np.sqrt(1/sum(1/(fit_i.intercept_stderr**2) for fit_i in fit_all_obj))

    #fig.add_trace(go.Scatter(
    #    mode='lines',
    #    x=list(line),
    #    y=list(map(lambda x: liniar_func(x, alpha_mean, beta_mean), line)),
    #    name=f'Averaged fit of all fits',
    #    hovertemplate=f'Averaged fit' + '<br>' + f'Slope: {alpha_mean:.3f} +- {alpha_stderr:.3f}' + '<br>' + f'Intercept: {beta_mean:.3f} +- {beta_stderr:.3f}' + '<br>',
    #    line=dict(color='DarkSlateGrey', )
    #))

    for oh_reac, ooh_reac in zip(oh_reactions, ooh_reactions):
        assert (metal := oh_reac.products[0].name.split('_')[0]) == ooh_reac.products[0].name.split('_')[0]
        for xc in functional_list:
            marker_arg = dict(marker=dict(size=16, color=colour_dict_metal[metal] if metal in colour_dict_metal.keys() else 'DarkSlateGrey', symbol=marker_dict_functional[xc.name] if xc.name in marker_dict_functional.keys() else 'circle', line=dict(color='Black', width=1)))
            try: fig.add_trace(go.Scatter(
                mode='markers',
                name=f'{xc.name}-{metal}',
                x=[(oh_adsorp := xc.calculate_reaction_enthalpy(oh_reac))],
                y=[xc.calculate_reaction_enthalpy(ooh_reac)],
                hovertemplate=f'metal: {metal}' + '<br>' + f'XC: {xc.name}' + '<br>' + f'OH adsorption: {str(oh_reac)}' + '   %{x:.3f}' + '<br>' + f'OOH adsorption: {str(ooh_reac)}' + '   %{y:.3f}',
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
                        y=(ens_y_cloud := xc.calculate_BEE_reaction_enthalpy(ooh_reac).tolist()),
                        x=(ens_x_cloud := xc.calculate_BEE_reaction_enthalpy(oh_reac).tolist()),
                        hovertemplate=f'metal: {metal}' + '<br>' + f'OH adsorption: {str(oh_reac)}' + '   %{x:.3f}' + '<br>' + f'OOH adsorption: {str(ooh_reac)}' + '   %{y:.3f}',
                        marker=dict(color=colour_dict_metal[metal] if metal in colour_dict_metal.keys() else 'Grey',
                                    opacity=0.5, ),
                        legendgroup=metal,
                        legendgrouptitle_text=metal,
                    ))

                    fig.update_traces(selector=dict(name=f'{xc.name}-{metal}'),
                                      error_x=dict(type='constant', value=sd(ens_x_cloud), color=colour_dict_metal[metal] if metal in colour_dict_metal.keys() else 'Grey', thickness=1.5, width=3, visible=False),
                                      error_y=dict(type='constant', value=sd(ens_y_cloud), color=colour_dict_metal[metal] if metal in colour_dict_metal.keys() else 'Grey', thickness=1.5, width=3, visible=False)
                                      )

                    fig.data = fig.data[-1:] + fig.data[0:-1]
                except: traceback.print_exc()

    if len(fig.data) > 0:
        min_value = min([min(fig.data, key=lambda d: d['x'])['x'], min(fig.data, key=lambda d: d['y'])['y']])[0]
        max_value = min([max(fig.data, key=lambda d: d['x'])['x'], max(fig.data, key=lambda d: d['y'])['y']])[0]

        fig.add_shape(type='line',
                      xref='x', yref='y',
                      x0=min_value, y0=min_value + 3.2,
                      x1=max_value, y1=max_value + 3.2,
                      line=dict(color='grey', width=3, dash='solid'),
                      opacity=0.5,
                      layer='below',
                      visible=True
                      )

    fig.update_layout(
        title='Scaling of OOH and OH',
        xaxis_title='OH adsorption energy',# in reference to Pt_{111} adsorption',
        yaxis_title='OOH adsorption energy',

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
            ),
            dict(
                type='buttons',
                direction='left',
                buttons=[
                    dict(
                        args=[{"visible": True}, [i for i, trace in enumerate(fig.data) if 'fit' in trace.name],
                              ],
                        label='Show all fits',
                        method='restyle',
                    ),
                    dict(
                        args=[{"visible": [True if match('linier scalling fit of .+', trace.name) or trace.name == f'Concatenated fit of all data points' else False if match('BEE fits No\. \d+ for .+', trace.name) else 'undefined' for i, trace in enumerate(fig.data) if 'fit' in trace.name]},
                              [i for i, trace in enumerate(fig.data) if 'fit' in trace.name]],
                        label='Show xc fits only',
                        method='restyle',
                    ),
                    dict(
                        args=[{"visible": False}, [i for i, trace in enumerate(fig.data) if match('linier scalling fit of .+', trace.name) or match('BEE fits No\. \d+ for .+', trace.name) or trace.name == f'Concatenated fit of all data points'],
                              ],
                        label='Hide all fits',
                        method='restyle',
                    ),
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.5,
                xanchor="left",
                y=1.1,
                yanchor="top"
            )
        ]
    )

    fig.set_subplots(rows=2, cols=1,row_heights=[0.7, 0.3])
    fig.add_traces(ens_figure.data, rows=2, cols=1)
    #fig.update_layout(**ens_figure.layout.__dict__, rows=2, cols=1)


    folder_exist('reaction_plots')
    save_name = 'reaction_plots/scaling_plot_OH_OOH_fits'
    if png_bool: fig.write_image(save_name + '.png')
    fig.write_html(save_name + '.html', include_mathjax='cdn')


def main(slab_db_dir: list[str], adsorbate_db_dir: list[str], mol_db_dir: list[str], thermo_dynamics: bool = False):
    pd_adsorbate_dat = build_pd(adsorbate_db_dir)
    pd_slab_dat = build_pd(slab_db_dir)
    pd_mol_dat = build_pd(mol_db_dir)

    functional_set = {xc for _, row in pd_adsorbate_dat.iterrows() if not pd.isna((xc := row.get('xc')))}

    oh_ad_h2_water = adsorption_OH_reactions[1::3]# metal_ref_ractions[0::2] #adsorption_OH_reactions[1::3] #[1,4,7,10,13,16]
    ooh_ad_h2_water = adsorption_OOH_reactions[1::3]#metal_ref_ractions[1::2] #adsorption_OOH_reactions[1::3]

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

