#partition=katla_verylong
#nprocshared=48
#mem=2000MB
#constrain='[v4|v5]'
import argparse
import os
import signal
from contextlib import contextmanager

from ase import Atoms
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


def mean(values): return sum(values) / len(values)
def sd(values, mean_value): return np.sqrt(1 / len(values) * sum(((x - mean_value) ** 2 for x in values)))


def sanitize(unclean_str: str) -> str:
    for ch in ['!', '*', '?', '{', '[', '(', ')', ']', '}', "'", '"']: unclean_str = unclean_str.replace(ch, '')
    for ch in ['/', '\\', '|', ' ', ',', '.']: unclean_str = unclean_str.replace(ch, '_')
    for ch in ['=', '+', ':', ';']: unclean_str = unclean_str.replace(ch, '-')
    return unclean_str



def single_point(image: Atoms, folder: str, name_tag: str, db_name: Optional[str] = None, spinpol: bool = False):
    grid_spacing = 0.16

    if spinpol:
        mag_value = 1
        mag_indicies = [atom.index for atom in image if atom.symbols in ['Ni', 'Co', 'Fe']]
        image.set_initial_magnetic_moments([mag_value if atom.index in mag_indicies else 0 for atom in image])

    calc = GPAW(mode=PW(500),
                kpts=(4, 4, 1),
                xc='BEEF-vdW',
                basis='dzp',
                txt=f'{folder}/BEE_image-{name_tag}.txt',
                gpts=h2gpts(grid_spacing, image.get_cell(), idiv=4),
                parallel={'augment_grids': True, 'sl_auto': True},
                convergence={'eigenstates': 0.000001},
                eigensolver=Davidson(3),
                spinpol=spinpol
                # hund=smile == 'O=O',
                # **setup_dic
                )

    image.set_calculator(calc)
    image.get_potential_energy()
    image.get_forces()
    ens = BEEFEnsemble(image)
    ensem_en_li = ens.get_ensemble_energies()
    ensem_mean = mean(ensem_en_li)
    ensem_sd = sd(ensem_en_li, ensem_mean)

    with db.connect(f'{sanitize(db_name)}.db') as db_obj:
        data_dict = {'ensemble_en': ensem_en_li}
        db_obj.write(image, name=f'{name_tag}', ensem_mean=ensem_mean, ensem_sd=ensem_sd, data=data_dict)

    del image, ens


def main(source: str, out_db: str, name_tag: Optional[str], spin: bool = False):
    folder = 'slabs'
    if world.rank == 0: folder_exist(os.path.basename(folder))
    slab = read(source)
    single_point(image=slab, folder=folder, name_tag=name_tag, db_name=out_db, spinpol=spin)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('out_db')
    parser.add_argument('--name_tag', '-n', default=None)
    parser.add_argument('--spin', action='store_true', default=None)
    args = parser.parse_args()

    main(**args.__dict__)