#partition=katla_medium
#nprocshared=8
#mem=2300MB
#constrain='[v4|v5]'
import argparse
import os
import ase.db as db
import numpy as np
from ase.io import read
from gpaw import GPAW, PW, Davidson
from gpaw.utilities import h2gpts
from typing import NoReturn, Sequence
from ase.parallel import parprint, world
from ase.dft.bee import BEEFEnsemble

def folder_exist(folder_name: str) -> NoReturn:
    if not  os.path.basename(folder_name) in os.listdir(os.path.dirname(folder_name)): os.mkdir(folder_name)

def sanitize(unclean_str: str) -> str:
    for ch in ['!', '*', '?', '{', '[', '(', ')', ']', '}',"'",'"']: unclean_str = unclean_str.replace(ch, '')
    for ch in ['/', '\\', '|',' ']: unclean_str = unclean_str.replace(ch, '_')
    for ch in ['=', '+', ':']: unclean_str = unclean_str.replace(ch, '-')
    return unclean_str

def weighted_mean(dat: Sequence[float|int], coef: Sequence[float|int]) -> float|int:
    return sum(dat_i*coef_i for dat_i,coef_i in zip(dat,coef))/ sum(coef)

def mean(values): return sum(values) / len(values)
def sd(values, mean_value): return np.sqrt(1 / len(values) * sum(((x - mean_value) ** 2 for x in values)))


def weighted_sd(dat: Sequence[float|int], coef: Sequence[float|int],mean:float|int) -> float|int:
    # https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf
    non_zero_coefs = len([co for co in coef if co != 0])
    return np.sqrt(
        sum(co_i * (dat_i - mean)**2 for co_i,dat_i in zip(coef,dat)) /
        (((non_zero_coefs-1)*sum(coef))/non_zero_coefs)
    )

def main(calc_name:str,structures:Sequence[str], db_name: str|None = None):


    folder = '/groups/kemi/thorkong/errors_investigation/transistion_calc/' + sanitize(calc_name)
    if world.rank == 0:
        folder_exist(folder)

    grid_spacing = 0.16
    for image_name in structures:
        image = read(image_name)
        calc = GPAW(mode=PW(500),
                    xc='RPBE',
                    basis='dzp',
                    txt=f'{folder}/BEE_{os.path.basename(image_name)}.txt',
                    gpts=h2gpts(grid_spacing, image.get_cell(), idiv=4),
                    parallel={'augment_grids': True, 'sl_auto': True},
                    convergence={'eigenstates': 0.000001},
                    eigensolver=Davidson(3),
                    #hund=smile == 'O=O',
                    #**setup_dic
                    )

        image.set_calculator(calc)
        potential_e = image.get_potential_energy()
        ens = BEEFEnsemble(image)
        ensem_en_li = ens.get_ensemble_energies()
        #ensem_coef = ens.get_beefvdw_ensemble_coefs()
        ensem_mean = mean(ensem_en_li)
        ensem_sd = sd(ensem_en_li,ensem_mean)

        if world.rank == 0:
            with db.connect(f'{folder}/{sanitize(db_name if db_name else calc_name)}.db') as db_obj:
                data_dict = {'ensemble_en': ensem_en_li} #,'ensemble_coef':ensem_coef}
                db_obj.write(image,ensem_mean = ensem_mean, ensem_sd = ensem_sd, data = data_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('calculation_name',type=str)
    parser.add_argument('files',nargs='+')
    parser.add_argument('--db_name','-n',type=str)
    args = parser.parse_args()

    main(args.calculation_name, args.files)