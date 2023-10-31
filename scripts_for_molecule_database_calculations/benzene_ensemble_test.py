import argparse
import sys
import pathlib
from typing import Sequence, Optional

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from scripts_for_molecule_database_calculations import build_pd, reaction, folder_exist

import numpy as np
import pandas as pd
from ase.db.core import bytes_to_object


class Functional:
    def __init__(self, functional_name: str,  mol_db: pd.DataFrame, needed_struc_dict: Optional[dict[str, list[str]]] = None, thermo_dynamic: bool = True):
        energy_type = 'enthalpy' if thermo_dynamic else 'energy'
        self.name = functional_name
        self.molecule = {smile: mol_db.query(f'smiles == "{smile}" and xc == "{functional_name}" and {energy_type}.notna()').get(energy_type).iloc[0] for smile in needed_struc_dict['molecule']}

        self.has_BEE = functional_name == 'BEEF-vdW'
        if self.has_BEE:
            self.molecule_energy = {smile: mol_db.query(f'smiles == "{smile}" and xc == "{functional_name}" and energy.notna()').get('energy').iloc[0] for smile in needed_struc_dict['molecule']}
            self.molecule_bee = {smile: np.array(bytes_to_object(mol_db.query(f'smiles == "{smile}" and xc == "{functional_name}" and _data.notna()').get('_data').iloc[0]).get('ensemble_en'))[:] for smile in needed_struc_dict['molecule']}
        else:
            self.molecule_energy = {}
            self.molecule_bee = {}

    def calculate_reaction_enthalpy(self, reaction_obj: reaction) -> float:
        reactant_enthalpy, product_enthalpy = tuple(sum(self.molecule[name] * amount for name, amount in getattr(reaction_obj, reac_part)) for reac_part in ('reactants', 'products'))
        return product_enthalpy - reactant_enthalpy

    def calculate_reaction_energy(self, reaction_obj: reaction) -> float:
        if not self.has_BEE: raise ValueError('calculate_reaction_energy only if the functional has BEE')
        reactant_enthalpy, product_enthalpy = tuple(sum(self.molecule_energy[name] * amount for typ, name, amount in getattr(reaction_obj, reac_part)) for reac_part in ('reactants', 'products'))
        return product_enthalpy - reactant_enthalpy

    def calculate_BEE_reaction_enthalpy(self, reaction_obj: reaction) -> np.ndarray[float]:
        if not self.has_BEE: raise ValueError('calculate_reaction_energy only if the functional has BEE')
        correction = self.calculate_reaction_enthalpy(reaction_obj) - self.calculate_reaction_energy(reaction_obj)
        reactant_BEE_enthalpy, product_BEE_enthalpy = tuple(sum(self.molecule_bee[name] * amount for typ, name, amount in getattr(reaction_obj, reac_part)) for reac_part in ('reactants', 'products'))
        return product_BEE_enthalpy - reactant_BEE_enthalpy + correction


def sd(values: Sequence[float], mean_value: Optional[float] = None) -> float:
    if not mean_value: mean_value = mean(values)
    return np.sqrt((1 / len(values)) * sum(((x - mean_value) ** 2 for x in values)))
def mean(values: Sequence[float]) -> float: return sum(values) / len(values)
def rsd(values: Sequence[float]) -> float: return sd(values, mean_value := mean(values)) / abs(mean_value)


def build_latex_sd_table(reaction_seq: Sequence[reaction], BEEF_vdW_functional: Functional):
    start_of_text = '\\begin{center}\n\\begin{tabular}{c|c|c|c}\n'
    end_of_text = '\\end{tabular}\n\\end{center}\n'

    first_line_text = '    Reaction & beef_vdw & $\\mu$ (eV) & $\\sigma$ \\\\ \n  \\hline &&\\\\ \n'

    main_text = '    '.join(f'{str(reac)} & {BEEF_vdW_functional.calculate_reaction_energy(reac):.3f}  &  {mean(BEEF_vdW_functional.calculate_BEE_reaction_enthalpy(reac)):.3f}  &  {sd(BEEF_vdW_functional.calculate_BEE_reaction_enthalpy(reac)):.3f}  \\\\ \n '
                + ('\\rowcolor{Gray} \n' * ((i+1) % 2))
                for i, reac in enumerate(reaction_seq))

    return start_of_text + first_line_text + main_text + end_of_text


def main(db_address: Sequence[str]):
    benz_pd = build_pd(db_address)

    benzene_reaction = reaction((('C1CCCCC1', 2),), (('C1CCCCC1.C1CCCCC1', 1),), None)

    folder_exist('reaction_plots')

    functional_list = [Functional(functional_name='BEEF-vdW', mol_db=benz_pd, needed_struc_dict=dict(molecule=['c1ccccc1', 'c1ccccc1.c1ccccc1']), thermo_dynamic=False)]

    with open('reaction_plots/Beef_ensemble_benzene_test.txt', 'w') as work_file:
        work_file.write(build_latex_sd_table((benzene_reaction,),functional_list[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db', nargs='+', help='path to one or more databases containing the data.')
    args = parser.parse_args()

    main(db_address=args.db)

