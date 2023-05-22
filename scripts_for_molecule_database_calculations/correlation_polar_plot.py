import ase.db as db
from ase.db.core import bytes_to_object
import os
import argparse
from dataclasses import dataclass, field
from typing import Sequence, NoReturn, Tuple, Iterable, Optional
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
#import plotly.express as px
import plotly.graph_objects as go


@dataclass
class reaction:
    reactants: Sequence[Tuple[str, int | float]]
    products: Sequence[Tuple[str, int | float]]
    experimental_ref: float

    def toStr(self) -> str:
        return ' + '.join([f'{n:.2f}{smi if smi != "cid281" else "C|||O"}' for smi, n in self.reactants]) + ' ---> ' + ' + '.join([f'{n:.2f}{smi  if smi != "cid281" else "C|||O"}' for smi, n in self.products])


@dataclass
class functional:
    name: str
    _correlation_vectors: Sequence[Tuple[float,float]] = field(default=None)

    @property
    def correlation_vectors(self):
        return self._correlation_vectors

    @correlation_vectors.setter
    def correlation_vectors(self, val: list[Tuple[float, float]]):
        if self._correlation_vectors is None: self._correlation_vectors = val
        else: self._correlation_vectors += val
#            for key in val.keys():
#                if key in self._correlation_vectors.keys(): self._correlation_vectors[key] += val[key]
#                else: self._correlation_vectors.update({key: val[key]})

    def calc_correlation_vector(self,reaction_1: reaction, reaction_2: reaction, dbo: db.core.Database | pd.DataFrame):

        reaction_1_val = reaction_enthalpy(reaction_1,self.name,dbo)
        reaction_2_val = reaction_enthalpy(reaction_2,self.name,dbo)

        correlation_vector = vector_minus([reaction_1.experimental_ref, reaction_2.experimental_ref], [reaction_1_val, reaction_2_val])
        self.correlation_vectors = [correlation_vector]


    def plotly_polar_plot(self):
        if self._correlation_vectors is not None:
            fig = go.Figure()

            cart_points = [cart_to_polar(point) for point in self._correlation_vectors]

            for cord in cart_points:
                fig.add_trace(go.Scatterpolar(
                    r=cord[0],
                    theta=cord[1],
                    mode='markers'
                ))

            fig.update_layout(
                title=dict(text=self.name),
                showlegend=False
            )

            fig.write_html('reaction_plots/' + f'{sanitize(self.name)}_polar_plot.html', include_mathjax='cdn')


def sanitize(unclean_str: str) -> str:
    for ch in ['!', '*', '?', '{', '[', '(', ')', ']', '}',"'",'"','.']: unclean_str = unclean_str.replace(ch, '')
    for ch in ['/', '\\', '|',' ']: unclean_str = unclean_str.replace(ch, '_')
    for ch in ['=', '+', ':',',']: unclean_str = unclean_str.replace(ch, '-')
    return unclean_str


def reaction_enthalpy(reac: reaction, functional: str, dbo: db.core.Database | pd.DataFrame, bee: bool = False) -> float | Tuple[float, float]:
    # exception for the wrong 37 row
    # if dbo.get([('smiles','=',smile),('xc','=',functional)]).get('id') != 37 else -21.630*amount
    if isinstance(dbo, db.core.Database):
        reac_enthalpy = sum(dbo.get([('smiles','=',smile),('xc','=',functional)]).get('enthalpy')*amount for smile,amount in reac.reactants)
        prod_enthalpy = sum(dbo.get([('smiles','=',smile),('xc','=',functional)]).get('enthalpy')*amount for smile,amount in reac.products)

        if bee and functional in ('BEEF-vdW',):  # ,"{'name':'BEEF-vdW','backend':'libvdwxc'}"):
            reac_ensamble_enthalpy = np.sum((np.array(dbo.get([('smiles','=',smile),('xc','=',functional)]).get('data').get('ensemble_en')[:]+(dbo.get([('smiles','=',smile),('xc','=',functional)]).get('enthalpy')-dbo.get([('smiles','=',smile),('xc','=',functional)]).get('energy')))*amount for smile,amount in reac.reactants),axis=0)
            prod_ensamble_enthalpy = np.sum((np.array(dbo.get([('smiles','=',smile),('xc','=',functional)]).get('data').get('ensemble_en')[:]+(dbo.get([('smiles','=',smile),('xc','=',functional)]).get('enthalpy')-dbo.get([('smiles','=',smile),('xc','=',functional)]).get('energy')))*amount for smile,amount in reac.products),axis=0)
            error_dev = (prod_ensamble_enthalpy-reac_ensamble_enthalpy).std()
            return error_dev, prod_enthalpy - reac_enthalpy
        return prod_enthalpy - reac_enthalpy

    elif isinstance(dbo, pd.DataFrame):
        reac_enthalpy = sum(dbo.query(f'smiles == "{smile}" and xc == "{functional}" and enthalpy.notna()').get('enthalpy').iloc[0]*amount for smile,amount in reac.reactants)
        prod_enthalpy = sum(dbo.query(f'smiles == "{smile}" and xc == "{functional}" and enthalpy.notna()').get('enthalpy').iloc[0]*amount for smile,amount in reac.products)

        if bee and functional in ('BEEF-vdW',):
            reac_ensamble_enthalpy = np.sum((np.array(bytes_to_object(dbo.query(f'smiles == "{smile}" and xc == "{functional}" and enthalpy.notna()').get('_data').iloc[0]).get('ensemble_en')[:]
                                                      +(dbo.query(f'smiles == "{smile}" and xc == "{functional}" and enthalpy.notna()').get('enthalpy').iloc[0]
                                                        -dbo.query(f'smiles == "{smile}" and xc == "{functional}" and enthalpy.notna()').get('energy').iloc[0]))*amount for smile,amount in reac.reactants),axis=0)
            prod_ensamble_enthalpy = np.sum((np.array(bytes_to_object(dbo.query(f'smiles == "{smile}" and xc == "{functional}" and enthalpy.notna()').get('_data').iloc[0]).get('ensemble_en')[:]
                                                      +(dbo.query(f'smiles == "{smile}" and xc == "{functional}" and enthalpy.notna()').get('enthalpy').iloc[0]
                                                      - dbo.query(f'smiles == "{smile}" and xc == "{functional}" and enthalpy.notna()').get('energy').iloc[0]))*amount for smile,amount in reac.products),axis=0)
            error_dev = (prod_ensamble_enthalpy-reac_ensamble_enthalpy).std()
            return error_dev, prod_enthalpy - reac_enthalpy
        return prod_enthalpy - reac_enthalpy
    raise ValueError('The type of database object was not recognised')


def vector_minus(v1: list[float], v2: list[float]) -> list[float]: return [a-b for a,b in zip(v1,v2)]
def cart_to_polar(cord: tuple[float,float]) -> tuple[float,float]: return np.sqrt(cord[0]**2+cord[1]**2), np.arctan2(cord[1],cord[2])

def main(db_dir: Sequence[str] = ('molreact.db',)):

    db_list = [db.connect(work_db) for work_db in db_dir]

    if len(db_list) == 1: pd_dat = db_list[0]
    else: pd_dat = pd.DataFrame([row.__dict__ for work_db in db_list for row in work_db.select()])

    functional_list = {xc for _, row in pd_dat.iterrows() if not pd.isna((xc := row.get('xc')))}
    functional_objs = [functional(func) for func in functional_list]

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
        reaction((('[HH]', 2), ('cid281', 1)), (('C=CC=C', 0.25), ('O', 1)), -1.08),  # 16
        reaction((('[HH]', 2), ('C(=O)=O', 1)), (('CC(O)=O', 0.5), ('O', 1)), -0.67),  # 17  a9
        reaction((('[HH]', 1), ('cid281', 1)), (('CC(O)=O', 0.5),), -1.1),  # 18
        reaction((('[HH]', 2), ('C(=O)=O', 1)), (('COC=O', 0.5),), -0.17),  # 19  a10
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
        reaction((('C1=CC=CC=C1', 1 / 6), ('O=O', 7 / 6)), (('C(=O)=O', 1), ('O', 0.5)), -33.83941093 / 6),  # 10
        reaction((('C1=CC=C(C=C1)O', 1 / 6), ('O=O', 7 / 6)), (('C(=O)=O', 1), ('O', 0.5)), -31.6325807 / 6),  # 10
    ]

    all_reactions = reactions + combustion_reactions

    for func_obj in functional_objs:
        for i in all_reactions:
            for j in all_reactions:
                if i != j:
                    try: func_obj.calc_correlation_vector(i, j, pd_dat)
                    except: print(f'correlation vector between reaction_1 ({i.toStr()}) and reaction_2 ({j.toStr()}) failed for reaction {func_obj.name}')
    for func_obj in functional_objs: func_obj.plotly_polar_plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db', nargs='+')
    args = parser.parse_args()

    main(args.db)
