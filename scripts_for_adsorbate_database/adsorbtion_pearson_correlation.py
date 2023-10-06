import argparse
import math
import sys
import pathlib
from typing import Sequence
from operator import itemgetter

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from scripts_for_adsorbate_database import sanitize, folder_exist, build_pd, adsorbate_reaction, all_adsorption_reactions
from scripts_for_adsorbate_database.adsorbate_correlation_plot import Functional
import plotly.graph_objects as go


def pearson(point_seq: Sequence[tuple[float, float]]) -> float:
    x_avg = sum(map(itemgetter(0), point_seq))/(n := len(point_seq))
    y_avg = sum(map(itemgetter(1), point_seq))/n
    return sum((x-x_avg)*(y-y_avg) for x, y in point_seq)/(math.sqrt(sum((x-x_avg)**2 for x, _ in point_seq))*math.sqrt(sum((y-y_avg)**2 for _, y in point_seq)))


def plot_correlation_matrix(reaction_seq: Sequence[adsorbate_reaction], BEEF_vdW_functional: Functional, png_bool: bool = False):
    fig = go.Figure()

    correlation_matrix = [[None]*(i+1)+[
        pearson(tuple(zip(BEEF_vdW_functional.calculate_BEE_reaction_enthalpy(reaction_1),
                          BEEF_vdW_functional.calculate_BEE_reaction_enthalpy(reaction_2))))
        for j, reaction_2 in enumerate(reaction_seq[i+1:], start=i+1)]
        for i, reaction_1 in enumerate(reaction_seq)]

    text_matrix = [[None]*(i+1)+[
        f'{str(reaction_1)}<br>{str(reaction_2)}'
        for j, reaction_2 in enumerate(reaction_seq[i+1:], start=i+1)]
        for i, reaction_1 in enumerate(reaction_seq)]

    fig.add_trace(go.Heatmap(
        z=correlation_matrix,
        #x=(text_axis:=[str(reac) for reac in reaction_seq]),#(nr_axis := [str(i) for i in range(len(reaction_seq))]),
        #y=text_axis, #nr_axis,
        text=text_matrix,
        #texttemplate='{text}',
        hoverongaps=False
    ))

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    folder_exist('reaction_plots')
    save_name = 'reaction_plots/' + f'correlation_matrix'
    if png_bool: fig.write_image(save_name + '.png')
    fig.write_html(save_name + '.html', include_mathjax='cdn')


def main(slab_db_dir: list[str], adsorbate_db_dir: list[str], mol_db_dir: list[str], png_bool: bool = False, reaction_list_bool: bool = False):
    pd_adsorbate_dat = build_pd(adsorbate_db_dir)
    pd_slab_dat = build_pd(slab_db_dir)
    pd_mol_dat = build_pd(mol_db_dir)

    #functional_set = {'BEEF-vdW'} # {xc for _, row in pd_adsorbate_dat.iterrows() if not pd.isna((xc := row.get('xc')))}

    dictionary_of_needed_strucs = {'molecule': [], 'slab': [], 'adsorbate': []}
    for reac in all_adsorption_reactions:
        for compo in reac.reactants + reac.products:
            dictionary_of_needed_strucs[compo.type].append(compo.name)

    functional_list = [Functional(functional_name='BEEF-vdW', slab_db=pd_slab_dat, adsorbate_db=pd_adsorbate_dat, mol_db=pd_mol_dat, needed_struc_dict=dictionary_of_needed_strucs, thermo_dynamic=False)]

    if reaction_list_bool:
        folder_exist('reaction_plots')
        with open('reaction_plots/reaction_lists.txt', 'w') as work_file:
            work_file.writelines([str(reac)+'\n' for reac in all_adsorption_reactions])

    plot_correlation_matrix(all_adsorption_reactions, functional_list[0], png_bool=png_bool)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-adb', '--adsorbate_db', nargs='+', help='path to one or more databases containing the data.')
    parser.add_argument('-mdb', '--molecule_db', nargs='+', help='path to one or more databases containing the data.')
    parser.add_argument('-sdb', '--slab_db', nargs='+', help='path to one or more databases containing the data.')
    #parser.add_argument('-m', '--metals', nargs='+', default=['Pt', 'Cu'])
    parser.add_argument('-png', '--png', action='store_true', default=False,)
    parser.add_argument('-list', '--reaction_list', action='store_true', default=False,)
    args = parser.parse_args()

    main(slab_db_dir=args.slab_db, adsorbate_db_dir=args.adsorbate_db, mol_db_dir=args.molecule_db, png_bool=args.png, reaction_list_bool=args.reaction_list)
