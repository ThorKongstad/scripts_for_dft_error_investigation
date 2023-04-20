#partition=katla_medium
#nprocshared=1
#mem=4000MB
#constrain='[v1|v2|v3|v4|v5]'
import argparse

import ase.db as db
import os
#import argparse
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



def reaction_error_plot(reac:reaction,funcionals:Sequence[str],dbo):
    fig, ax = plt.subplots(figsize=(9, 4))

    for func in funcionals:
        ax.scatter(x=reaction_enthalpy(reac, func, dbo) - reac.experimental_ref, y=0, label=func)

    ax.legend()
    ax.set_title(reac.toStr())
    ax.set_ylim(-0.01,0.05)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel(f'deviation from reference: {reac.experimental_ref}ev')

    folder_exist('reaction_plots')
    fig.save('reaction_plots/'+reac.toStr()+'.png')

def total_plot(reac_list: Sequence[reaction],funcionals:Sequence[str],dbo,add_name=''):
    list_colours = ['tab:blue','tab:orange','tab:green','tab:red']
    heights = [0,0.005,0.01,0.015]

    if add_name == '': head_react= ['C(=O)=O','C|||O']
    else: head_react= ['C(=O)=O']

    for cur_head_react in head_react:
        fig, ax = plt.subplots(figsize=(16, 4))
        for c_nr,func in enumerate(funcionals):
#            for reac in reac_list:
#                if cur_head_react in reac.toStr():
            ax.scatter(*zip(*[(reaction_enthalpy(reac, func, dbo) - reac.experimental_ref, heights[c_nr]) for reac in reac_list if cur_head_react in reac.toStr()]), label=func, alpha = 0.3, marker='D', c=list_colours[c_nr], s=700)
        ax.legend()
        ax.set_title(f'{cur_head_react} {add_name}')
        ax.set_ylim(-0.01, 0.05)
        ax.get_yaxis().set_visible(False)
        ax.set_xlabel(f'deviation from reference in ev')

        ax.grid()

        folder_exist('reaction_plots')
        fig.savefig('reaction_plots/' + f'all_reactions_with_{cur_head_react}_{add_name}.png')

def get_dat(reactions,functionals,dbo):
    for reac in reactions:
        print(reac.toStr() + '  ' + ';  '.join((f'{func}  {reaction_enthalpy(reac, func, dbo) - reac.experimental_ref}' for func in functionals)))

    for func in functionals:
        print(f'{func} mean error:  {sum(fabs(reaction_enthalpy(reac, func, dbo) - reac.experimental_ref) for reac in reactions) / len(reactions)}')

def main(db_dir: str = 'molreact.db'):

    if not os.path.basename(db_dir) in os.listdir(os.path.dirname(db_dir)): raise FileNotFoundError("Can't find database")

    db_obj = db.connect(db_dir)
    functionals = ('PBE','RPBE','BEEF-vdW',"{'name':'BEEF-vdW','backend':'libvdwxc'}")

#    {'CCC', 'O', 'C=CC=C', 'C', 'cid281', 'CC(C)O', 'C1CCCCC1', 'O=CO', 'OO', 'COC', 'C1=CC=C(C=C1)O', 'CCO', 'C(=O)=O', 'O=O', 'COC=O', 'CCCC', 'C1=CC=CC=C1', 'CC(O)=O', 'C=C', 'CC', '[HH]', 'CO'}

    reactions = [
        reaction((('[HH]',1),('C(=O)=O',1)),(('cid281',1),('O',1)),0.43), #0
        reaction((('[HH]',4),('C(=O)=O',1)),(('C',1),('O',2)),-1.71), #1
        reaction((('[HH]',3),('cid281',1)),(('C',1),('O',1)),-2.14), #2
        reaction((('[HH]',1),('C(=O)=O',1)),(('O=CO',1),),0.15), #3
        reaction((('cid281',1),('O',1)),(('O=CO',1),),-0.27), #4
        reaction((('[HH]',3),('C(=O)=O',1)),(('CO',1),('O',1)),-0.55), #5
        reaction((('[HH]',2),('cid281',1)),(('CO',1),),-0.98), #6
        reaction((('[HH]',3),('C(=O)=O',1)),(('CCO',0.5),('O',1.5)),-0.89), #7
        reaction((('[HH]',2),('cid281',1)),(('CCO',0.5),('O',0.5)),-1.32), #8
        reaction((('[HH]', 10/3), ('C(=O)=O', 1)),(('CCC',1/3),('O',2)),-1.3), #9
        reaction((('[HH]',7/3),('cid281',1)),(('CCC',1/3),('O',1)),-1.72), #10
        reaction((('[HH]',7/2),('C(=O)=O',1)),(('CC',1/2),('O',2)),-1.37), #11
        reaction((('[HH]',2.5),('cid281',1)),(('CC',0.5),('O',1)),-1.8), #12
        reaction((('[HH]', 3), ('C(=O)=O', 1)),(('C=C',1/2),('O',2)),-0.66), #13
        reaction((('[HH]', 2), ('cid281', 1)),(('C=C',0.5),('O',1)),-1.09), #14
        reaction((('[HH]', 2.75), ('C(=O)=O', 1)),(('C=CC=C',1/4),('O',2)),-0.65), #15
        reaction((('[HH]', 2), ('cid281', 1)),(('C=CC=C',0.25),('O',1)),-1.08), #16
        reaction((('[HH]', 2), ('C(=O)=O', 1)),(('CC(O)=O',0.5),('O',1)),-0.67), #17
        reaction((('[HH]', 1), ('cid281', 1)),(('CC(O)=O',0.5),),-1.1), #18
        reaction((('[HH]', 2), ('C(=O)=O', 1)),(('COC=O',0.5),),-0.17), #19
        reaction((('[HH]', 1), ('cid281', 1)), (('COC=O', 0.5),), -0.60)  #20
    ]

    combustion_reactions =[
        reaction((('CC',1/2),('O=O',(7/2)/2)),(('C(=O)=O',1),('O',1.5)),-16.15491767/2), #1
        reaction((('CCC',1/3),('O=O',5/3)),(('C(=O)=O',1),('O',4/3)),-22.99618466/3), #2
        reaction((('CCCC', 1/4), ('O=O', (13/2)/4)), (('C(=O)=O', 1), ('O', 5/4)), -29.79920448/4),  #3
        reaction((('CO', 1), ('O=O', 3/2)), (('C(=O)=O', 1), ('O', 2)), -7.529923227),  #4
        reaction((('CCO', 0.5), ('O=O', 1.5)), (('C(=O)=O', 1), ('O', 1.5)), -14.16108994/2),  #5
        reaction((('CC(C)O', 1/3), ('O=O', (9/2)/3)), (('C(=O)=O', 1), ('O', 4/3)), -20.78116602/3),  #6
        reaction((('COC', 1/2), ('O=O', 1.5)), (('C(=O)=O', 1), ('O', 1.5)), -15.13001808/2),  #7
        reaction((('O=CO', 1), ('O=O', 1/2)), (('C(=O)=O', 1), ('O', 1)), -2.637499504),  #8
        reaction((('CC(O)=O', 0.5), ('O=O', 1)), (('C(=O)=O', 1), ('O', 1)), -9.059602457/2),  #9
        reaction((('C1CCCCC1', 1/6), ('O=O', 9/6)), (('C(=O)=O', 1), ('O', 1)), -40.60200692/6),  #10
        reaction((('C1=CC=CC=C1', 1/6), ('O=O', 7/6)), (('C(=O)=O', 1), ('O', 0.5)), -33.83941093/6),  # 10
        reaction((('C1=CC=C(C=C1)O', 1/6), ('O=O', 7/6)), (('C(=O)=O', 1), ('O', 0.5)), -31.6325807/6),  # 10
    ]

    if False:
        for rea in reactions:
            try:
                reaction_error_plot(rea, functionals, db_obj)
            except:
                continue

    if True:
        total_plot(reactions+combustion_reactions,functionals,db_obj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-db','--database',help='directory to the database, if not stated will look for molreact.db in pwd.', default='molreact.db')
    args = parser.parse_args()

    main(args.database)
