import argparse
import os
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Sequence, NoReturn, Tuple, Iterable, Optional
from re import match
from operator import attrgetter

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from scripts_for_molecule_database_calculations import build_pd, all_reactions

import pandas as pd
import plotly.graph_objects as go



@dataclass
class sic_functional:
    name: str
    molecule_dict: dict[str, float]

    def __post_init__(self):
        if self.name == 'PBE': self.sic_amount = 0
        elif (amount_match := match(r'PBE-PZ-SIC-direct-(?P<amount>\d+)', self.name)): self.sic_amount = float(amount_match.group('amount')) / 100
        else: self.sic_amount = 0.5

    def calc_reaction(self, reaction_obj: 'reaction', correction_dict: Optional[dict[str,float]] = None):
        if correction_dict is None: correction_dict = {}
        reactants_enthalpy = sum((self.molecule_dict[reactant] + (0 if reactant not in correction_dict.keys() else correction_dict.get(reactant)))*amount for reactant, amount in reaction_obj.reactants)
        product_enthalpy = sum((self.molecule_dict[product] + (0 if product not in correction_dict.keys() else correction_dict.get(product)))*amount for product, amount in reaction_obj.products)

        return product_enthalpy - reactants_enthalpy


def plot_sic_deviation(functional_obj_seq: Sequence[sic_functional], reaction_seq: Sequence['reaction']):
    fig = go.Figure()

    functional_obj_seq_sorted = sorted(functional_obj_seq, key=attrgetter('sig_amount'))

    for reac in reaction_seq:
        try:
            template_str = reac.toStr()
            colour = 'darkviolet' if ('O=O' in template_str and 'C|||O' in template_str) else (
                'firebrick' if 'O=O' in template_str else (
                'royalblue' if 'C|||O' in template_str else 'black'))

            fig.add_trace(go.Scatter(
                x= tuple(func.sic_amount for func in functional_obj_seq_sorted),
                y=tuple(func.calc_reaction(reac) - reac.experimental_ref for func in functional_obj_seq_sorted),
                mode='markers+line',
                hovertemplate=template_str,
                marker=dict(color=colour, size=16),
                line=dict(color=colour,),
                opacity=0.4
            ))
        except: pass

    fig.update_layout(
        title=dict(text='Sic deviation'),
        showlegend=False,
        xaxis_title='Deviation from experimental reference',
        yaxis_title='sic %'
    )

    fig.write_html('reaction_plots/' + f'sic_deviation_plot.html', include_mathjax='cdn')


def main(sic_db_dir: str, pbe_db_dir: str):

    sic_pd = build_pd(sic_db_dir)
    pbe_pd = build_pd(pbe_db_dir, 'xc=PBE')

    functional_list = {xc for _, row in sic_pd.iterrows() if not pd.isna((xc := row.get('xc'))) and match('PBE-PZ-SIC-direct(-(?P<amount>\d+))?',xc)}
    functional_objs = [sic_functional('PBE', {smile: enthalpy for _, row in pbe_pd.iterrows() if row.get('xc') == 'PBE' and not pd.isna((smile := row.get('smiles'))) and not pd.isna((enthalpy := row.get('enthalpy')))})] \
                      + [sic_functional(func, {smile: enthalpy for _, row in sic_pd.iterrows() if row.get('xc') == func and not pd.isna((smile := row.get('smiles'))) and not pd.isna((enthalpy := row.get('enthalpy')))}) for func in functional_list]

    plot_sic_deviation(functional_objs, all_reactions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sic_db')
    parser.add_argument('pbe_db')
    args = parser.parse_args()

    main(args.sic_db, args.pbe_db)