import argparse
import math
import sys
import pathlib
from typing import Sequence, Optional
from contextlib import suppress

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from scripts_for_adsorbate_database import sanitize, folder_exist, build_pd, adsorbate_reaction, adsorption_OH_reactions, adsorption_OOH_reactions, metal_ref_ractions
from scripts_for_adsorbate_database.adsorbate_correlation_plot import Functional

import numpy as np


def sd(values: Sequence[float], mean_value: Optional[float] = None) -> float:
    if not mean_value: mean_value = mean(values)
    return np.sqrt((1 / len(values)) * sum(((x - mean_value) ** 2 for x in values)))
def mean(values: Sequence[float]) -> float: return sum(values) / len(values)
def rsd(values: Sequence[float]) -> float: return sd(values, mean_value := mean(values)) / abs(mean_value)


def build_latex_sd_table(reaction_seq: Sequence[adsorbate_reaction], BEEF_vdW_functional: Functional):
    start_of_text = '\\begin{center}\n\\begin{tabular}{c|c|c|c}\n'
    end_of_text = '\\end{tabular}\n\\end{center}\n'

    first_line_text = '    Reaction & beef_vdw & $\\mu$ (eV) & $\\sigma$ \\\\ \n  \\hline &&\\\\ \n'

    #main_text = '    '.join(f'{str(reac)} & {BEEF_vdW_functional.calculate_reaction_energy(reac)}  &  {mean(BEEF_vdW_functional.calculate_BEE_reaction_enthalpy(reac)):.3f}  &  {sd(BEEF_vdW_functional.calculate_BEE_reaction_enthalpy(reac)):.3f}  \\\\ \n '
    #            + ('\\rowcolor{Gray} \n' * ((i+1) % 2))
    #            for i, reac in enumerate(reaction_seq))
    main_text = ''

    for i, reac in enumerate(reaction_seq):
        try: main_text += '    ' + f'{str(reac)} & {BEEF_vdW_functional.calculate_reaction_energy(reac)}  &  {mean(BEEF_vdW_functional.calculate_BEE_reaction_enthalpy(reac)):.3f}  &  {sd(BEEF_vdW_functional.calculate_BEE_reaction_enthalpy(reac)):.3f}  \\\\ \n ' + ('\\rowcolor{Gray} \n' * ((i+1) % 2))
        except: pass

    return start_of_text + first_line_text + main_text + end_of_text


def main(slab_db_dir: list[str], adsorbate_db_dir: list[str], mol_db_dir: list[str], reaction_list_bool: bool = False):
    pd_adsorbate_dat = build_pd(adsorbate_db_dir)
    pd_slab_dat = build_pd(slab_db_dir)
    pd_mol_dat = build_pd(mol_db_dir)

    dictionary_of_needed_strucs = {'molecule': [], 'slab': [], 'adsorbate': []}
    for reac in adsorption_OH_reactions + adsorption_OOH_reactions + metal_ref_ractions:
        for compo in reac.reactants + reac.products:
            dictionary_of_needed_strucs[compo.type].append(compo.name)

    functional_list = [Functional(functional_name='BEEF-vdW', slab_db=pd_slab_dat, adsorbate_db=pd_adsorbate_dat, mol_db=pd_mol_dat, needed_struc_dict=dictionary_of_needed_strucs, thermo_dynamic=False)]

    folder_exist('reaction_plots')

    if reaction_list_bool:
        with open('reaction_plots/reaction_lists.txt', 'w') as work_file:
            work_file.writelines([str(reac)+'\n' for reac in adsorption_OH_reactions + adsorption_OOH_reactions + metal_ref_ractions])

    with open('reaction_plots/Beef_ensemble_sd.txt', 'w') as work_file:
        work_file.writelines('\n\n'.join(build_latex_sd_table(reac_seq, functional_list[0]) for reac_seq in [adsorption_OH_reactions, adsorption_OOH_reactions, metal_ref_ractions]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-adb', '--adsorbate_db', nargs='+', help='path to one or more databases containing the data.')
    parser.add_argument('-mdb', '--molecule_db', nargs='+', help='path to one or more databases containing the data.')
    parser.add_argument('-sdb', '--slab_db', nargs='+', help='path to one or more databases containing the data.')
    parser.add_argument('-list', '--reaction_list', action='store_true', default=False,)
    args = parser.parse_args()

    main(slab_db_dir=args.slab_db, adsorbate_db_dir=args.adsorbate_db, mol_db_dir=args.molecule_db, reaction_list_bool=args.reaction_list)
