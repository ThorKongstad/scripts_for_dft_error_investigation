import ase.db as db
import argparse
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Sequence, NoReturn, Tuple, Iterable, Optional
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from scripts_for_molecule_database_calculations import build_pd, all_reactions, sanitize
import pandas as pd
import plotly.graph_objects as go


@dataclass
class reaction:
    reactants: Sequence[Tuple[str, float]]
    products: Sequence[Tuple[str, float]]
    experimental_ref: float

    def toStr(self) -> str:
        return ' + '.join([f'{n:.2g}{smi if smi != "cid281" else "C|||O"}' for smi, n in self.reactants]) + ' ---> ' + ' + '.join([f'{n:.2g}{smi  if smi != "cid281" else "C|||O"}' for smi, n in self.products])


@dataclass
class functional:
    name: str
    molecule_dict: dict[str, float]

    def calc_reaction(self, reaction_obj: reaction, correction_dict: Optional[dict[str, float]] = None):
        if correction_dict is None: correction_dict = {}
        reactants_enthalpy = sum((self.molecule_dict[reactant] + (0 if reactant not in correction_dict.keys() else correction_dict.get(reactant)))*amount for reactant, amount in reaction_obj.reactants)
        product_enthalpy = sum((self.molecule_dict[product] + (0 if product not in correction_dict.keys() else correction_dict.get(product)))*amount for product, amount in reaction_obj.products)

        return product_enthalpy - reactants_enthalpy


def plot_correction(functional_obj: functional, reaction_seq: Sequence[reaction], correction_dict: Optional[dict[str, float]] = None):
    fig = go.Figure()

    for reac in reaction_seq:
        try:
            template_str = reac.toStr()
            colour = 'darkviolet' if ('O=O' in template_str and 'C|||O' in template_str) else (
                'firebrick' if 'O=O' in template_str else (
                'royalblue' if 'C|||O' in template_str else 'black'))

            fig.add_trace(go.Scatter(
                x=[reac.experimental_ref],
                y=[functional_obj.calc_reaction(reac,correction_dict)],
                mode='markers',
                hovertemplate=template_str,
                marker=dict(color=colour, size=16)
            ))
        except: pass

    if len(fig.data) > 0:
        min_value = min([min(fig.data, key=lambda d: d['x'])['x'], min(fig.data, key=lambda d: d['y'])['y']])
        max_value = min([max(fig.data, key=lambda d: d['x'])['x'], max(fig.data, key=lambda d: d['y'])['y']])

        fig.add_shape(type='line',
                      xref="x",yref='y',
                      x0=min_value, y0=min_value,
                      x1=max_value, y1=max_value,
                      line=dict(color='grey', width=3, dash='solid'),
                      opacity=0.5,
                      layer='below',
                      visible=True
                      )

    fig.update_layout(
        title=dict(text=functional_obj.name),
        showlegend=False,
        xaxis_title='Experimental reference',
        yaxis_title='Calculated enthalpy'
    )

    fig.write_html('reaction_plots/' + f'{sanitize(functional_obj.name)}_correction_plot.html', include_mathjax='cdn')


def main(db_dir: Sequence[str] = ('molreact.db',)):

    db_list = [db.connect(work_db) for work_db in db_dir]

    if False: pass #len(db_list) == 1: pd_dat = db_list[0]
    else: pd_dat = pd.DataFrame([row.__dict__ for work_db in db_list for row in work_db.select()])

    functional_list = {xc for _, row in pd_dat.iterrows() if not pd.isna((xc := row.get('xc')))}
    functional_objs = [functional(func, {smile: enthalpy for _, row in pd_dat.iterrows() if row.get('xc') == func and not pd.isna((smile := row.get('smiles'))) and not pd.isna((enthalpy := row.get('enthalpy')))}) for func in functional_list]

    reactions = [
        reaction((('[HH]', 1), ('C(=O)=O', 1)), (('cid281', 1), ('O', 1)), 0.43),  # 0  a0
        reaction((('[HH]', 4), ('C(=O)=O', 1)), (('C', 1), ('O', 2)), -1.71),  # 1  a1
        reaction((('[HH]', 3), ('cid281', 1)), (('C', 1), ('O', 1)), -2.14),  # 2
        reaction((('[HH]', 1), ('C(=O)=O', 1)), (('O=CO', 1),), 0.15),  # 3  a2
        reaction((('cid281', 1), ('O', 1)), (('O=CO', 1),), -0.27),  # 4
        reaction((('[HH]', 3), ('C(=O)=O', 1)), (('CO', 1), ('O', 1)), -0.55),  # 5  a3
        reaction((('[HH]', 2), ('cid281', 1)), (('CO', 1),), -0.98),  # 6
        reaction((('[HH]', 3), ('C(=O)=O', 1)), (('CCO', 0.5), ('O', 1.5)), -0.89),  # 7  a4
        reaction((('[HH]', 2), ('cid281', 1)), (('CCO', 0.5), ('O', 0.5)), -1.32),  # 8
        reaction((('[HH]', 10 / 3), ('C(=O)=O', 1)), (('CCC', 1 / 3), ('O', 2)), -1.3),  # 9  a5
        reaction((('[HH]', 7 / 3), ('cid281', 1)), (('CCC', 1 / 3), ('O', 1)), -1.72),  # 10
        reaction((('[HH]', 7 / 2), ('C(=O)=O', 1)), (('CC', 1 / 2), ('O', 2)), -1.37),  # 11 a6
        reaction((('[HH]', 2.5), ('cid281', 1)), (('CC', 0.5), ('O', 1)), -1.8),  # 12
        reaction((('[HH]', 3), ('C(=O)=O', 1)), (('C=C', 1 / 2), ('O', 2)), -0.66),  # 13  a7
        reaction((('[HH]', 2), ('cid281', 1)), (('C=C', 0.5), ('O', 1)), -1.09),  # 14
        reaction((('[HH]', 2.75), ('C(=O)=O', 1)), (('C=CC=C', 1 / 4), ('O', 2)), -0.65),  # 15  a8
        reaction((('[HH]', 1.75), ('cid281', 1)), (('C=CC=C', 0.25), ('O', 1)), -1.08),  # 16
        reaction((('[HH]', 2), ('C(=O)=O', 1)), (('CC(O)=O', 0.5), ('O', 1)), -0.67),  # 17  a9
        reaction((('[HH]', 1), ('cid281', 1)), (('CC(O)=O', 0.5),), -1.1),  # 18
        reaction((('[HH]', 2), ('C(=O)=O', 1)), (('COC=O', 0.5), ('O', 1)), -0.17),  # 19  a10
        reaction((('[HH]', 1), ('cid281', 1)), (('COC=O', 0.5),), -0.60)  # 20
    ]

    varification_reactions = [
        reaction((('[HH]', 3), ('C(=O)=O', 1)), (('COC', 0.5), ('O', 4 / 3)), None),  # a15
    ]

    combustion_reactions = [
        reaction((('CC', 1 / 2), ('O=O', (7 / 2) / 2)), (('C(=O)=O', 1), ('O', 1.5)), -16.15491767 / 2),  # 1
        reaction((('CCC', 1 / 3), ('O=O', 5 / 3)), (('C(=O)=O', 1), ('O', 4 / 3)), -22.99618466 / 3),  # 2
        reaction((('CCCC', 1 / 4), ('O=O', (13 / 2) / 4)), (('C(=O)=O', 1), ('O', 5 / 4)), -29.79920448 / 4),  # 3
        reaction((('CO', 1), ('O=O', 3 / 2)), (('C(=O)=O', 1), ('O', 2)), -7.529923227),  # 4
        reaction((('CCO', 0.5), ('O=O', 1.5)), (('C(=O)=O', 1), ('O', 1.5)), -14.16108994 / 2),  # 5
        reaction((('CC(C)O', 1 / 3), ('O=O', (9 / 2) / 3)), (('C(=O)=O', 1), ('O', 4 / 3)), -20.78116602 / 3),  # 6
        reaction((('COC', 1 / 2), ('O=O', 1.5)), (('C(=O)=O', 1), ('O', 1.5)), -15.13001808 / 2),  # 7
        reaction((('O=CO', 1), ('O=O', 1 / 2)), (('C(=O)=O', 1), ('O', 1)), -2.637499504),  # 8
        reaction((('CC(O)=O', 0.5), ('O=O', 1)), (('C(=O)=O', 1), ('O', 1)), -9.059602457 / 2),  # 9
        reaction((('C1CCCCC1', 1 / 6), ('O=O', 9 / 6)), (('C(=O)=O', 1), ('O', 1)), -40.60200692 / 6),  # 10
        reaction((('C1=CC=CC=C1', 1 / 6), ('O=O', 1.25)), (('C(=O)=O', 1), ('O', 0.5)), -33.83941093 / 6),  # 11
        reaction((('C1=CC=C(C=C1)O', 1 / 6), ('O=O', 7 / 6)), (('C(=O)=O', 1), ('O', 0.5)), -31.6325807 / 6),  # 12
    ]

    oxygen_reactions = [
        reaction((('[HH]',1),('O=O',0.5)),(('O',1),), None),
        reaction((('[HH]', 1), ('O=O', 1)), (('OO', 1),), None),
    ]

    all_reactions = reactions + combustion_reactions

    for func_obj in functional_objs:
        plot_correction(func_obj, all_reactions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db', nargs='+', help='path to one or more databases containing the data.')
    args = parser.parse_args()

    main(args.db)