#partition=katla_verylong
#nprocshared=48
#mem=2000MB
#constrain='[v4|v5]'
import argparse
import os
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from sqlite3 import OperationalError
import ase.db as db
import numpy as np
from ase.io import read
from ase.optimize import QuasiNewton
from gpaw import GPAW, PW, Davidson
from gpaw.utilities import h2gpts
from typing import NoReturn, Sequence, Optional
from ase.parallel import parprint, world, barrier
from ase.dft.bee import BEEFEnsemble
import time


@retry(retry=retry_if_exception_type(FileExistsError), stop=stop_after_attempt(5), wait=wait_fixed(2))
def folder_exist(folder_name: str, path: str = '.') -> None:
    if folder_name not in os.listdir(path): os.mkdir(ends_with(path, '/') + folder_name)


def ends_with(string: str, end_str: str) -> str:
    return string + end_str * (end_str != string[-len(end_str):0])


def sanitize(unclean_str: str) -> str:
    for ch in ['!', '*', '?', '{', '[', '(', ')', ']', '}', "'", '"']: unclean_str = unclean_str.replace(ch, '')
    for ch in ['/', '\\', '|', ' ', ',', '.']: unclean_str = unclean_str.replace(ch, '_')
    for ch in ['=', '+', ':', ';']: unclean_str = unclean_str.replace(ch, '-')
    return unclean_str


def weighted_mean(dat: Sequence[float], coef: Sequence[float]) -> float:
    return sum(dat_i*coef_i for dat_i, coef_i in zip(dat, coef)) / sum(coef)


def mean(values): return sum(values) / len(values)
def sd(values, mean_value): return np.sqrt(1 / len(values) * sum(((x - mean_value) ** 2 for x in values)))


def weighted_sd(dat: Sequence[float], coef: Sequence[float], mean: float) -> float:
    # https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf
    non_zero_coefs = len([co for co in coef if co != 0])
    return np.sqrt(
        sum(co_i * (dat_i - mean)**2 for co_i, dat_i in zip(coef, dat)) /
        (((non_zero_coefs-1)*sum(coef))/non_zero_coefs)
    )


@retry(retry=retry_if_exception_type(OperationalError), stop=stop_after_attempt(5), wait=wait_fixed(10))
def update_db(db_dir: str, db_update_args: dict):
    with db.connect(db_dir) as db_obj:
        db_obj.update(**db_update_args)


def main(calc_name: str, structures: Sequence[str], db_name: Optional[str] = None, direc: Optional[str] = '.'):
    folder = ends_with(direc, '/') + sanitize(calc_name)
    if world.rank == 0: folder_exist(os.path.basename(folder), path=os.path.dirname(folder))

    grid_spacing = 0.16

    if len(structures) == 1: images = read(structures[0], index=':')
    else: images = map(read, structures)

    for nr, image in enumerate(images):
        barrier()
        calc = GPAW(mode=PW(500),
                    kpts=(4, 4, 1),
                    xc='BEEF-vdW',
                    basis='dzp',
                    txt=f'{folder}/BEE_image-{nr}.txt',
                    gpts=h2gpts(grid_spacing, image.get_cell(), idiv=4),
                    parallel={'augment_grids': True, 'sl_auto': True},
                    convergence={'eigenstates': 0.000001},
                    eigensolver=Davidson(3),
                    #hund=smile == 'O=O',
                    #**setup_dic
                    )

        image.set_calculator(calc)
        image.get_potential_energy()
        image.get_forces()
        ens = BEEFEnsemble(image)
        ensem_en_li = ens.get_ensemble_energies()
        ensem_mean = mean(ensem_en_li)
        ensem_sd = sd(ensem_en_li, ensem_mean)

        if world.rank == 0:
            data_dict = {'ensemble_en': ensem_en_li}
            db_obj = db.connect(f'{folder}/{sanitize(db_name if db_name else calc_name)}.db')
            db_obj.write(image, name=f'image_{nr}', ensem_mean=ensem_mean, ensem_sd=ensem_sd, data=data_dict)

            # CONTEXT MANAGER CURRENTLY MAKING TROUBLE.
#            with db.connect(f'{folder}/{sanitize(db_name if db_name else calc_name)}.db') as db_obj:
#                data_dict = {'ensemble_en': ensem_en_li}
#                db_obj.write(image, name=f'image_{nr}', ensem_mean=ensem_mean, ensem_sd=ensem_sd, data=data_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('calculation_name', type=str)
    parser.add_argument('files', nargs='+')
    parser.add_argument('--db_name', '-n', type=str)
    #parser.add_argument('-opt', '--optimise', action='store_true')
    parser.add_argument('-dir', '--directory', type=str, default='.')
    args = parser.parse_args()

    main(args.calculation_name, args.files, args.db_name, args.directory)
