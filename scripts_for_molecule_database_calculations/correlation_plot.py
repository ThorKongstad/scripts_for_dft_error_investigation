import ase.db as db
from ase.db.core import bytes_to_object
import os
import argparse
from dataclasses import dataclass
from typing import Sequence, NoReturn, Tuple, Iterable, Optional
#from math import fabs
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
        return ' + '.join([f'{n:.2g}{smi if smi != "cid281" else "C|||O"}' for smi,n in self.reactants]) + ' ---> ' + ' + '.join([f'{n:.2g}{smi  if smi != "cid281" else "C|||O"}' for smi,n in self.products])


def folder_exist(folder_name: str, path: str = '.', tries: int = 10) -> NoReturn:
    try:
        tries -= 1
        if folder_name not in os.listdir(path): os.mkdir(ends_with(path, '/')+folder_name)
    except FileExistsError:
        time.sleep(2)
        if tries > 0: folder_exist(folder_name, path=path, tries=tries)


def ends_with(string: str, end_str: str) -> str:
    return string + end_str * (end_str != string[-len(end_str):0])


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


def BEE_reaction_enthalpy(reac:reaction, functional: str, dbo: db.core.Database | pd.DataFrame) -> np.ndarray:
    if isinstance(dbo, db.core.Database):
        reac_ensamble_enthalpy = np.sum((np.array(
            dbo.get([('smiles', '=', smile), ('xc', '=', functional)]).get('data').get('ensemble_en')[:] + (
                        dbo.get([('smiles', '=', smile), ('xc', '=', functional)]).get('enthalpy') - dbo.get(
                    [('smiles', '=', smile), ('xc', '=', functional)]).get('energy'))) * amount for smile, amount in
                                         reac.reactants), axis=0)
        prod_ensamble_enthalpy = np.sum((np.array(
            dbo.get([('smiles', '=', smile), ('xc', '=', functional)]).get('data').get('ensemble_en')[:] + (
                        dbo.get([('smiles', '=', smile), ('xc', '=', functional)]).get('enthalpy') - dbo.get(
                    [('smiles', '=', smile), ('xc', '=', functional)]).get('energy'))) * amount for smile, amount in
                                         reac.products), axis=0)
        return reac_ensamble_enthalpy - prod_ensamble_enthalpy

    elif isinstance(dbo, pd.DataFrame):
        reac_ensamble_enthalpy = np.sum(list((
            np.array(bytes_to_object(dbo.query(f'smiles == "{smile}" and xc == "{functional}" and enthalpy.notna()').get('_data').iloc[0]).get('ensemble_en')[:])
            + (dbo.query(f'smiles == "{smile}" and xc == "{functional}" and enthalpy.notna()').get('enthalpy').iloc[0]
            - dbo.query(f'smiles == "{smile}" and xc == "{functional}" and enthalpy.notna()').get('energy').iloc[0])) * amount
                                      for smile, amount in reac.reactants), axis=0)
        prod_ensamble_enthalpy = np.sum(list((
            np.array(bytes_to_object(dbo.query(f'smiles == "{smile}" and xc == "{functional}" and enthalpy.notna()').get('_data').iloc[0]).get('ensemble_en')[:])
            + (dbo.query(f'smiles == "{smile}" and xc == "{functional}" and enthalpy.notna()').get('enthalpy').iloc[0]
            - dbo.query(f'smiles == "{smile}" and xc == "{functional}" and enthalpy.notna()').get('energy').iloc[0])) * amount for smile, amount in reac.products), axis=0)
        return reac_ensamble_enthalpy - prod_ensamble_enthalpy
    raise ValueError('The type of database object was not recognised')


def BEE_reaction_enthalpy_final_energy_correction(reac: reaction, functional: str, dbo: db.core.Database | pd.DataFrame) -> np.ndarray:
    if isinstance(dbo, db.core.Database):
        correction = reaction_enthalpy(reac, functional, dbo) \
                     -(sum(dbo.get([('smiles','=',smile),('xc','=',functional)]).get('energy')*amount for smile,amount in reac.reactants)
                     -sum(dbo.get([('smiles','=',smile),('xc','=',functional)]).get('energy')*amount for smile,amount in reac.products))
        reac_ensamble_enthalpy = sum(np.array(
            dbo.get([('smiles', '=', smile), ('xc', '=', functional)]).get('data').get('ensemble_en'))[:] * amount for smile, amount in
                                         reac.reactants)
        prod_ensamble_enthalpy = sum(np.array(
            dbo.get([('smiles', '=', smile), ('xc', '=', functional)]).get('data').get('ensemble_en'))[:] * amount for smile, amount in
                                         reac.products)
        return reac_ensamble_enthalpy - prod_ensamble_enthalpy + correction

    elif isinstance(dbo, pd.DataFrame):
        correction = (reaction_enthalpy(reac, functional, dbo)
                     -(sum(dbo.query(f'smiles == "{smile}" and xc == "{functional}" and enthalpy.notna()').get('energy').iloc[0]*amount for smile,amount in reac.reactants)
                     -sum(dbo.query(f'smiles == "{smile}" and xc == "{functional}" and enthalpy.notna()').get('energy').iloc[0]*amount for smile,amount in reac.products)))
        reac_ensamble_enthalpy = sum(np.array(
            bytes_to_object(dbo.query(f'smiles == "{smile}" and xc == "{functional}" and enthalpy.notna()').get('_data').iloc[0]).get('ensemble_en'))[:] * amount for smile, amount in reac.reactants)
        prod_ensamble_enthalpy = sum(np.array(
            bytes_to_object(dbo.query(f'smiles == "{smile}" and xc == "{functional}" and enthalpy.notna()').get('_data').iloc[0]).get('ensemble_en'))[:] * amount for smile, amount in reac.products)
        return reac_ensamble_enthalpy - prod_ensamble_enthalpy + correction
    raise ValueError('The type of database object was not recognised')


def correlation_plot(reaction_1: reaction, reaction_2: reaction, dbo: db.core.Database | pd.DataFrame, reaction_indexes: Optional[Tuple[int, int]] = None):
    fig, ax = plt.subplots()

    colour_dict = {
        'PBE': 'indianred',
        'RPBE': 'firebrick',
        'PBE-PZ-SIC': 'darkorange',
        'BEEF-vdW': 'mediumblue',
        "{'name':'BEEF-vdW','backend':'libvdwxc'}": 'mediumpurple'
    }

    if isinstance(dbo, db.core.Database): functional_list = {row.get('xc') for row in dbo.select()}
    elif isinstance(dbo, pd.DataFrame): functional_list = {xc for _, row in dbo.iterrows() if not pd.isna((xc := row.get('xc')))}
    else: raise ValueError('The type of database object was not recognised')

    for c_nr, func in enumerate(functional_list):
        colour_arg = {'color':colour_dict[func]} if func in colour_dict.keys() else {}
        try: ax.scatter(x=reaction_enthalpy(reaction_1, func, dbo), y=reaction_enthalpy(reaction_2, func, dbo), label=func, zorder=c_nr+1, **colour_arg)
        except: pass
        if func == 'BEEF-vdW':
            try: ax.scatter(x=BEE_reaction_enthalpy_final_energy_correction(reaction_1, func, dbo).tolist(), y=BEE_reaction_enthalpy_final_energy_correction(reaction_2, func, dbo).tolist(), label=f'BEE for {func}', c='grey', alpha=0.2, zordor=0)
            except: pass
    ax.scatter(x=reaction_1.experimental_ref,y=reaction_2.experimental_ref,label='experimental ref',marker='X',c='gold')

#    ax.legend()
    if isinstance(reaction_indexes,tuple): ax.set_title(f'correlation between reaction {reaction_indexes[0]} and {reaction_indexes[1]}')
    else: ax.set_title(f'correlation between two reactions')
    ax.set_title(f'correlation between two reactions')
    ax.set_xlabel(reaction_1.toStr())
    ax.set_ylabel(reaction_2.toStr())

    folder_exist('reaction_plots')
    if isinstance(reaction_indexes, tuple): fig.savefig('reaction_plots/' + f'correlation_plot_{"-".join([str(x) for x in reaction_indexes])}.png')
    else: fig.savefig('reaction_plots/correlation_plot.png')


def correlation_plotly(reaction_1: reaction, reaction_2: reaction, dbo: db.core.Database | pd.DataFrame, reaction_indexes: Optional[Tuple[int, int]] = None):
    fig = go.Figure()

    colour_dict = {
        'PBE': 'idianred',
        'RPBE': 'firebrick',
        'PBE-PZ-SIC': 'darkorange',
        'BEEF-vdW': 'mediumblue',
        "{'name':'BEEF-vdW','backend':'libvdwxc'}": 'mediumpurple'
    }

    functionals = {xc for _, func in dbo.iterrows() if not pd.isna((xc := func.get('xc')))}

    for c_nr, func in enumerate(functionals):
        marker_arg = dict(marker={'color':colour_dict[func]}) if func in colour_dict.keys() else {}
        try:
            fig.add_trace(go.Scatter(
             x=reaction_enthalpy(reaction_1, func, dbo),
             y=reaction_enthalpy(reaction_2, func, dbo),
             name=func,
             mode='markers',
             **marker_arg))
            if func == 'BEEF-vdW':
                try:
                    ensamble_trace = go.Scatter(
                        x=BEE_reaction_enthalpy_final_energy_correction(reaction_1, func, dbo).tolist(),
                        y=BEE_reaction_enthalpy_final_energy_correction(reaction_2, func, dbo).tolist(),
                        name=f'BEE for {func}',
                        mode='markers',
                        marker=dict(color='Grey',opacity=0.5,)
                    )
                    fig.data = (ensamble_trace,) + fig.data
                except: pass
        except: continue

    fig.update_layout(
        xaxis_title=reaction_1.toStr(),
        yaxis_title=reaction_2.toStr()
    )

    folder_exist('reaction_plots')
    if reaction_indexes: save_name = 'reaction_plots/' + f'correlation_plot_{"-".join([str(x) for x in reaction_indexes])}.html'
    else: save_name = 'reaction_plots/correlation_plot.html'
    fig.write_html(save_name, include_mathjax='cdn')


def sp(x):  # silent print
    print(x)
    return x


def main(reaction_index_1: int, reaction_index_2: int, db_dir: Sequence[str] = ('molreact.db',)):
    #if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir)) > 0 else '.'): raise FileNotFoundError("Can't find database")
    #db_obj = db.connect(db_dir)
    #functionals = {row.get('xc') for row in db_obj.select()}
    #functionals = ('PBE', 'RPBE', 'BEEF-vdW', "{'name':'BEEF-vdW','backend':'libvdwxc'}")

    db_list = [db.connect(work_db) for work_db in db_dir]

    if len(db_list) == 1: pd_dat = db_list[0]
    else: pd_dat = pd.DataFrame([row.__dict__ for work_db in db_list for row in work_db.select()])

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
        reaction((('[HH]', 3), ('C(=O)=O', 1)), (('COC', 0.5), ('O', 4/3)), None),  # a15
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
        reaction((('C1=CC=CC=C1', 1 / 6), ('O=O', 7 / 6)), (('C(=O)=O', 1), ('O', 0.5)), -33.83941093 / 6),  # 11
        reaction((('C1=CC=C(C=C1)O', 1 / 6), ('O=O', 7 / 6)), (('C(=O)=O', 1), ('O', 0.5)), -31.6325807 / 6),  # 12
    ]

    # a total of 33 reactions

    all_reactions = reactions + combustion_reactions

    correlation_plot(all_reactions[reaction_index_1], all_reactions[reaction_index_2], pd_dat, (reaction_index_1, reaction_index_2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('reaction_1',type=int)
    parser.add_argument('reaction_2',type=int)
    parser.add_argument('-db','--database',help='directory to the database, if not stated will look for molreact.db in pwd.', default=('molreact.db',), nargs='+')
    args = parser.parse_args()

    main(args.reaction_1, args.reaction_2, args.database)

