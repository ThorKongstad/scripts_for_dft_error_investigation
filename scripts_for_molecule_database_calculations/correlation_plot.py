import ase.db as db
import os
import argparse
from dataclasses import dataclass
from typing import Sequence, NoReturn, Tuple
from math import fabs
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class reaction:
    reactants: Sequence[Tuple[str,int|float]]
    products: Sequence[Tuple[str,int|float]]
    experimental_ref: float

    def toStr(self) -> str:
        return ' + '.join([f'{n}{smi if smi != "cid281" else "C|||O"}' for smi,n in self.reactants]) + ' ---> ' + ' + '.join([f'{n}{smi  if smi != "cid281" else "C|||O"}' for smi,n in self.products])

def folder_exist(folder_name: str, path:str = '.') -> NoReturn:
    if folder_name not in os.listdir(path): os.mkdir(ends_with(path, '/')+folder_name)

def ends_with(st:str, end:str) -> str:
    if st[-1] != end: return st+end
    else: return st

def reaction_enthalpy(reac:reaction, functional:str, dbo=None,bee:bool = False) -> float|Tuple[float,float]:
    if dbo == None:dbo = db.connect('/groups/kemi/thorkong/errors_investigation/molreact.db')

    # exception for the wrong 37 row
    # if dbo.get([('smiles','=',smile),('xc','=',functional)]).get('id') != 37 else -21.630*amount

    reac_enthalpy = sum(dbo.get([('smiles','=',smile),('xc','=',functional)]).get('enthalpy')*amount for smile,amount in reac.reactants)
    prod_enthalpy = sum(dbo.get([('smiles','=',smile),('xc','=',functional)]).get('enthalpy')*amount for smile,amount in reac.products)

    if bee and functional in ('BEEF-vdW'):#,"{'name':'BEEF-vdW','backend':'libvdwxc'}"):
        reac_ensamble_enthalpy = np.sum((np.array(dbo.get([('smiles','=',smile),('xc','=',functional)]).get('data').get('ensemble_en')[:]+(dbo.get([('smiles','=',smile),('xc','=',functional)]).get('enthalpy')-dbo.get([('smiles','=',smile),('xc','=',functional)]).get('energy')))*amount for smile,amount in reac.reactants),axis=0)
        prod_ensamble_enthalpy = np.sum((np.array(dbo.get([('smiles','=',smile),('xc','=',functional)]).get('data').get('ensemble_en')[:]+(dbo.get([('smiles','=',smile),('xc','=',functional)]).get('enthalpy')-dbo.get([('smiles','=',smile),('xc','=',functional)]).get('energy')))*amount for smile,amount in reac.products),axis=0)
        error_dev = (prod_ensamble_enthalpy-reac_ensamble_enthalpy).std()
        return error_dev, prod_enthalpy - reac_enthalpy
    return prod_enthalpy - reac_enthalpy

def BEE_reaction_enthalpy(reac:reaction, functional:str, dbo=None)-> np.ndarray:
    if dbo == None:dbo = db.connect('/groups/kemi/thorkong/errors_investigation/molreact.db')
    reac_ensamble_enthalpy = sum((np.array(
        dbo.get([('smiles', '=', smile), ('xc', '=', functional)]).get('data').get('ensemble_en')[:] + (
                    dbo.get([('smiles', '=', smile), ('xc', '=', functional)]).get('enthalpy') - dbo.get(
                [('smiles', '=', smile), ('xc', '=', functional)]).get('energy'))) * amount for smile, amount in
                                     reac.reactants))
    prod_ensamble_enthalpy = sum((np.array(
        dbo.get([('smiles', '=', smile), ('xc', '=', functional)]).get('data').get('ensemble_en')[:] + (
                    dbo.get([('smiles', '=', smile), ('xc', '=', functional)]).get('enthalpy') - dbo.get(
                [('smiles', '=', smile), ('xc', '=', functional)]).get('energy'))) * amount for smile, amount in
                                     reac.products))
    return reac_ensamble_enthalpy-prod_ensamble_enthalpy

def BEE_reaction_enthalpy_final_energy_correction(reac:reaction, functional:str, dbo=None)-> np.ndarray:
    if dbo == None:dbo = db.connect('/groups/kemi/thorkong/errors_investigation/molreact.db')
    correction = reaction_enthalpy(reac, functional, dbo) \
                 -(sum(dbo.get([('smiles','=',smile),('xc','=',functional)]).get('energy')*amount for smile,amount in reac.reactants)
                 -sum(dbo.get([('smiles','=',smile),('xc','=',functional)]).get('energy')*amount for smile,amount in reac.products))
    reac_ensamble_enthalpy = sum(np.array(
        dbo.get([('smiles', '=', smile), ('xc', '=', functional)]).get('data').get('ensemble_en'))[:] * amount for smile, amount in
                                     reac.reactants)
    prod_ensamble_enthalpy = sum(np.array(
        dbo.get([('smiles', '=', smile), ('xc', '=', functional)]).get('data').get('ensemble_en'))[:] * amount for smile, amount in
                                     reac.products)
    return reac_ensamble_enthalpy-prod_ensamble_enthalpy + correction

def correlation_plot(reaction_1:reaction,reaction_2:reaction,dbo,functional_list:Sequence[str],reaction_indexes:Tuple[int,int]|None=None):
    fig, ax = plt.subplots()

    for c_nr, func in enumerate(functional_list):
        ax.scatter(x=reaction_enthalpy(reaction_1,func,dbo),y=reaction_enthalpy(reaction_2,func,dbo),label= func)
        if func == 'BEEF-vdW':
            try:
                ax.scatter(x=BEE_reaction_enthalpy(reaction_1,func,dbo).tolist(),y=BEE_reaction_enthalpy(reaction_2,func,dbo).tolist(),label=f'BEE for {func}',c='grey',alpha=0.2)
            except: pass
    ax.scatter(x=reaction_1.experimental_ref,y=reaction_2.experimental_ref,label='experimental ref',marker='X',c='firebrick')

    ax.legend()
    if isinstance(reaction_indexes,tuple): ax.set_title(f'correlation between reaction {reaction_indexes[0]} and {reaction_indexes[1]}')
    else: ax.set_title(f'correlation between two reactions')
    ax.set_title(f'correlation between two reactions')
    ax.set_xlabel(reaction_1.toStr())
    ax.set_ylabel(reaction_2.toStr())

    folder_exist('reaction_plots')
    if isinstance(reaction_indexes, tuple): fig.savefig('reaction_plots/' + f'correlation_plot_{"-".join([str(x) for x in reaction_indexes])}.png')
    else: fig.savefig('reaction_plots/correlation_plot.png')


def main(reaction_index_1:int,reaction_index_2:int,db_dir: str = 'molreact.db'):
    if not os.path.basename(db_dir) in os.listdir(os.path.dirname(db_dir)): raise FileNotFoundError("Can't find database")
    db_obj = db.connect(db_dir)
    functionals = ('PBE', 'RPBE', 'BEEF-vdW', "{'name':'BEEF-vdW','backend':'libvdwxc'}")

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
        reaction((('[HH]',3),('C(=O)=O', 1)),(('COC',0.5),('O',4/3)),None) # a15
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

    correlation_plot(all_reactions[reaction_index_1],all_reactions[reaction_index_2],db_obj,functionals,(reaction_index_1,reaction_index_2))

if __name__ == '__mmain__':
    parser = argparse.ArgumentParser()
    parser.add_argument('reaction_1',type=int)
    parser.add_argument('reaction_2',type=int)
    parser.add_argument('-db','--database',help='directory to the database, if not stated will look for molreact.db in pwd.', default='molreact.db')
    args = parser.parse_args()

    main(args.reaction_1, args.reaction_2, args.database)

