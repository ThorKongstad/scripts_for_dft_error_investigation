import argparse
import math
import sys
import pathlib
from typing import Sequence, Optional
import traceback
from re import match
from dataclasses import dataclass, field

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from scripts_for_adsorbate_database import sanitize, folder_exist, build_pd, adsorbate_reaction, adsorption_OH_reactions, adsorption_OOH_reactions, metal_ref_ractions, adsorption_O_reactions, sd, mean
from scripts_for_adsorbate_database.adsorbate_correlation_plot import Functional

import numpy as np
import pandas as pd
from scipy import stats, odr
import plotly.graph_objects as go
import plotly.express as px


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


def scaling_fit(reac_1_Energies: Sequence[float], reac_2_Energies: Sequence[float], reac_1_Energies_sigma: Optional[Sequence[float]] = None, reac_2_Energies_sigma: Optional[Sequence[float]] = None):
    if any(sigma_val is not None for sigma_val in (reac_1_Energies_sigma, reac_2_Energies_sigma)):
        @dataclass
        class odr_linear_fitting_results:
            slope: float
            stderr: float
            intercept: float
            intercept_stderr: float

        fitting_model = odr.Model(lambda beta, x: Linear_func(x, a=1, b=beta[0]))

        fit_odr_dat = odr.Data(x=reac_1_Energies, wd=1 / (np.power(reac_1_Energies_sigma, 2)), y=reac_2_Energies, we=1 / (np.power(reac_2_Energies_sigma, 2)))
        odr_fit_obj = odr.ODR(fit_odr_dat, fitting_model, beta0=[3.2]).run()  # beta0 is the initial guess for the scalling relation
        fit_result = odr_linear_fitting_results(slope=1, intercept=odr_fit_obj.beta[0], stderr=0, intercept_stderr=odr_fit_obj.sd_beta[0])
    else:
        fit_result = stats.linregress(x=reac_1_Energies, y=reac_2_Energies)
    return fit_result


def Linear_func(x: float, a: float, b: float) -> float: return a*x+b


def overpotential(dG_OOH: float, dG_OH: float, dG_O: float) -> float: return min((4.92 - dG_OOH, dG_OOH - dG_O, dG_O - dG_OH, dG_OH)) # 1.23 -


def vulcano_plotly(functional_list: Sequence[Functional], oh_reactions: Sequence[adsorbate_reaction], ooh_reactions: Sequence[adsorbate_reaction], o_reactions: Sequence[adsorbate_reaction], oh_reaction_relative: Sequence[adsorbate_reaction], ooh_reactions_relative: Sequence[adsorbate_reaction], o_reactions_relative: Sequence[adsorbate_reaction],  png_bool: bool = False):
    fig = go.Figure()

    colour_dict_functional = {
        'PBE': 'indianred',
        'RPBE': 'firebrick',
        'PBE-PZ-SIC': 'darkorange',
        'BEEF-vdW': 'mediumblue',
        "{'name':'BEEF-vdW','backend':'libvdwxc'}": 'mediumpurple'
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

    OH_corr = 0.35 - 0.5 # + 0.35 is dZPE - TdS from 10.1021/jp047349j, - 0.5 is water stability correction 10.1021/cs300227s
    OOH_corr = 0.4 - 0.3

    beef = [xc for xc in functional_list if xc.name == 'BEEF-vdW'][0]

    OH_OOH_scalling_fit = ode_1par_linear([beef.calculate_reaction_enthalpy(OH_reac) + OH_corr for OH_reac in oh_reactions],
                                      [beef.calculate_reaction_enthalpy(OOH_reac) + OOH_corr for OOH_reac in ooh_reactions],
                                      [sd(beef.calculate_BEE_reaction_enthalpy(OH_reac) + OH_corr) for OH_reac in oh_reactions],
                                      [sd(beef.calculate_BEE_reaction_enthalpy(OOH_reac) + OOH_corr) for OOH_reac in ooh_reactions])

    oh_ensamble = list(map(lambda reac: beef.calculate_BEE_reaction_enthalpy(reac) + OH_corr, oh_reactions)) # is a nested matrix like object, with rows corresponding the metals and col as each ensamble function
    ooh_ensamble = list(map(lambda reac: beef.calculate_BEE_reaction_enthalpy(reac) + OOH_corr, ooh_reactions))
    fit_ens_objs = [ode_1par_linear(reac_1_Energies=OH_vals, reac_2_Energies=OOH_vals) for OH_vals, OOH_vals in zip(zip(*oh_ensamble), zip(*ooh_ensamble))]#[stats.linregress(x=OH_vals, y=OOH_vals) for OH_vals, OOH_vals in zip(zip(*oh_ensamble), zip(*ooh_ensamble))]
    intercept_ens = np.array(tuple(fit.intercept for fit in fit_ens_objs))

    beta_G_mean = mean(intercept_ens)
    beta_G_SD = sd(intercept_ens, beta_G_mean)

    volcano_peak_ens = (4.92 - intercept_ens)/2
    volcano_peak_mean = (4.92 - beta_G_mean) / 2
    volcano_peak_sd = beta_G_SD/2

    #for i, ens_peak in enumerate(volcano_peak_ens):
    #    if i % 100 == 0: print(f'plotting volcano ens peaks {i} out of 2000')
    #    fig.add_vline(
    #        x=ens_peak,
    #        line_color='#0000CD',
    #        opacity=0.2
    #    )

    fig.add_vline(
        x=volcano_peak_mean,
        line_dash='dash',
    )

    fig.add_vrect(
        x0=volcano_peak_mean - volcano_peak_sd, x1=volcano_peak_mean + volcano_peak_sd,
        fillcolor="green",
        opacity=0.25,
        annotation_text=f"V0lcano top location {volcano_peak_mean} <br> stderr: {volcano_peak_sd:.3f} eV", annotation_position="top left"
    )

    print('plotting Pt')
    beef_pt_OH = volcano_peak_mean - 0.11
    fig.add_trace(go.Scatter(
        mode='markers',
        name=f'BEEF-vdW-Pt',
        x=[beef_pt_OH],
        y=[beef_pt_OH],
        hovertemplate=f'functional: BEEF-vdW' + '<br>' + f'metal: Pt' + '<br>',
        error_x=dict(type='constant', value=volcano_peak_sd,
                     color=colour_dict_metal['Pt'], thickness=1.5,
                     width=3, visible=False),
        legendgroup='Pt',
        legendgrouptitle_text='Pt',
        marker=dict(size=16, color=colour_dict_metal['Pt'], symbol=marker_dict_functional['BEEF-vdW'])
    ))

    fig.add_trace(go.Scatter(
        mode='markers',
        name=f'BEE for Pt BEEF-vdW',
        y=volcano_peak_ens - 0.11,
        x=volcano_peak_ens - 0.11,
        hovertemplate=f'metal: Pt',
        marker=dict(color=colour_dict_metal['Pt'], opacity=0.5, ),
        legendgroup='Pt',
        legendgrouptitle_text='Pt',
    ))

    for oh_reac, ooh_reac in zip(oh_reaction_relative, ooh_reactions_relative,):
        assert (metal := oh_reac.products[0].name.split('_')[0]) == ooh_reac.products[0].name.split('_')[0]
        print(f'plotting {metal}')
        xc = beef
        marker_arg = dict(marker=dict(size=16, color=colour_dict_metal[metal] if metal in colour_dict_metal.keys() else 'DarkSlateGrey', symbol=marker_dict_functional[xc.name] if xc.name in marker_dict_functional.keys() else 'circle'))
        try: fig.add_trace(go.Scatter(
            mode='markers',
            name=f'{xc.name}-{metal}',
            x=[(OH_ad := beef_pt_OH + xc.calculate_reaction_enthalpy(oh_reac))], # + 0.35 is dZPE - TdS from 10.1021/jp047349j, - 0.3 is water stability correction 10.1021/cs300227s
            y=[overpotential(
                dG_OOH=(ooh_adsorp := OH_ad + beta_G_mean),
                dG_OH=OH_ad,
                dG_O=OH_ad * 2 # 0.05 is dZPE - TdS from 10.1021/acssuschemeng.8b04173
            )],
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
                    x=(ens_x_cloud := (volcano_peak_ens - 0.11) + xc.calculate_BEE_reaction_enthalpy(oh_reac)),
                    y=(ens_y_cloud := np.array(list(map(lambda ooh, oh, o: overpotential(
                            dG_OOH=ooh,
                            dG_OH=oh,
                            dG_O=o
                            ),
                        (ens_x_cloud + intercept_ens).tolist(),
                        (ens_x_cloud).tolist(),
                        (ens_x_cloud * 2).tolist()
                        )))),
                    hovertemplate=f'metal: {metal}' + '<br>' + f'OH adsorption: {str(oh_reac)}' + '<br>' + f'OOH adsorption: {str(ooh_reac)}',
                    marker=dict(color=colour_dict_metal[metal] if metal in colour_dict_metal.keys() else 'Grey', opacity=0.5, ),
                    legendgroup=metal,
                    legendgrouptitle_text=metal,
                ))

                fig.update_traces(selector=dict(name=f'{xc.name}-{metal}'),
                                  error_x=dict(type='constant', value=sd(ens_x_cloud), color=colour_dict_metal[metal] if metal in colour_dict_metal.keys() else 'Grey', thickness=1.5, width=3, visible=False),
                                  error_y=dict(type='constant', value=sd(ens_y_cloud), color=colour_dict_metal[metal] if metal in colour_dict_metal.keys() else 'Grey', thickness=1.5, width=3, visible=False)
                                  )

                fig.data = fig.data[-1:] + fig.data[0:-1]
            except: pass

    fig.update_layout(
        title='ORR',
        xaxis_title='$\Delta G_{*OH} - \Delta G_{Pt111*OH}$',# in reference to Pt_{111} adsorption',
        yaxis_title='Limiting potential relative to Pt',

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
                               'error_y.visible': [True if match('BEEF-vdW-[A   -Z][a-z]', trace.name) else False for trace in fig.data]
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
    save_name = 'reaction_plots/vulcano_plot_relative'
    if png_bool: fig.write_image(save_name + '.png')
    fig.write_html(save_name + '.html', include_mathjax='cdn')


def main(slab_db_dir: list[str], adsorbate_db_dir: list[str], mol_db_dir: list[str], thermo_dynamics: bool = False):
    pd_adsorbate_dat = build_pd(adsorbate_db_dir)
    pd_slab_dat = build_pd(slab_db_dir)
    pd_mol_dat = build_pd(mol_db_dir)

    functional_set = {xc for _, row in pd_adsorbate_dat.iterrows() if not pd.isna((xc := row.get('xc')))}

    oh_ad_metal_ref = metal_ref_ractions[0::3] #adsorption_OH_reactions[1::3] #[1,4,7,10,13,16]
    oh_ad_h2_water = adsorption_OH_reactions[1::3] #[1,4,7,10,13,16]

    o_ad_h2_water = adsorption_O_reactions[1::3] #[1,4,7,10,13,16]

    ooh_ad_metal_ref = metal_ref_ractions[1::3] #adsorption_OOH_reactions[1::3]
    ooh_ad_h2_water = adsorption_OOH_reactions[1::3]

    dictionary_of_needed_strucs = {'molecule': [], 'slab': [], 'adsorbate': []}
    for reac in oh_ad_h2_water + ooh_ad_h2_water + o_ad_h2_water:
        for compo in reac.reactants + reac.products:
            dictionary_of_needed_strucs[compo.type].append(compo.name)

    functional_list = []
    for xc in functional_set:
        try: functional_list.append(Functional(functional_name=xc, slab_db=pd_slab_dat, adsorbate_db=pd_adsorbate_dat, mol_db=pd_mol_dat, needed_struc_dict=dictionary_of_needed_strucs, thermo_dynamic=thermo_dynamics))
        except: pass

    vulcano_plotly(functional_list,
                   oh_reactions=oh_ad_h2_water,
                   ooh_reactions=ooh_ad_h2_water,
                   o_reactions=o_ad_h2_water,
                   oh_reaction_relative=oh_ad_metal_ref,
                   ooh_reactions_relative=ooh_ad_metal_ref,
                   o_reactions_relative=o_ad_h2_water)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-adb', '--adsorbate_db', nargs='+', help='path to one or more databases containing the data.')
    parser.add_argument('-mdb', '--molecule_db', nargs='+', help='path to one or more databases containing the data.')
    parser.add_argument('-sdb', '--slab_db', nargs='+', help='path to one or more databases containing the data.')
    args = parser.parse_args()

    main(slab_db_dir=args.slab_db, adsorbate_db_dir=args.adsorbate_db, mol_db_dir=args.molecule_db)
