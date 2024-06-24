import argparse
import os
from typing import NoReturn, Sequence, Tuple, Never, Optional, NamedTuple
#from dataclasses import dataclass, field
#from itertools import chain
#from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
import numpy as np
import ase.db as db
import pandas as pd

from ase.db.core import bytes_to_object


def build_pd(db_dir_list, select_key: Optional = None):
    if isinstance(db_dir_list, str): db_dir_list = [db_dir_list]
    for db_dir in db_dir_list:
        if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir)) > 0 else '.'):
            raise FileNotFoundError("Can't find database")
    db_list = [db.connect(work_db) for work_db in db_dir_list]
    pd_dat = pd.DataFrame([row.__dict__ for work_db in db_list for row in work_db.select(selection=select_key)])
    return pd_dat


def mean(values): return sum(values) / len(values)
def sd(values, mean_value): return np.sqrt(1 / len(values) * sum(((x - mean_value) ** 2 for x in values)))


def main(neb_image_1: str, slab_for_1: str, neb_image_2: str, slab_for_2: str):

    neb_1_adr = neb_image_1.split('@')
    slab_1_adr = slab_for_1.split('@')
    neb_2_adr = neb_image_2.split('@')
    slab_2_adr = slab_for_2.split('@')

    neb_1_pd = build_pd(neb_1_adr[0], select_key=(neb_1_adr[1] if len(neb_1_adr) > 1 else None))
    slab_1_pd = build_pd(slab_1_adr[0], select_key=(slab_1_adr[1] if len(slab_1_adr) > 1 else None))
    neb_2_pd = build_pd(neb_2_adr[0], select_key=(neb_2_adr[1] if len(neb_2_adr) > 1 else None))
    slab_2_pd = build_pd(slab_2_adr[0], select_key=(slab_2_adr[1] if len(slab_2_adr) > 1 else None))

    energy_diff = neb_2_pd.iloc[0].loc['energy'] + slab_1_pd.iloc[0].loc['energy'] - neb_1_pd.iloc[0].loc['energy'] - slab_2_pd.iloc[0].loc['energy']

    neb_1_ens = bytes_to_object(neb_1_pd.iloc[0].loc['_data']).get('ensemble_en')
    slab_1_ens = bytes_to_object(slab_1_pd.iloc[0].loc['_data']).get('ensemble_en')
    neb_2_ens = bytes_to_object(neb_2_pd.iloc[0].loc['_data']).get('ensemble_en')
    slab_2_ens = bytes_to_object(slab_2_pd.iloc[0].loc['_data']).get('ensemble_en')

    energy_ensemble = [n_2 + s_2 - n_1 - s_1 for n_1, s_1, n_2, s_2 in zip(neb_1_ens, slab_1_ens, neb_2_ens, slab_2_ens)]

    print(f'energy difference between image 1 and the reference 2 is: {energy_diff:.3f} +- {sd(energy_ensemble, mean_value=mean(energy_ensemble)):.3f} eV')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('neb_image_1', help='format: db_path.db@row_nr')
    parser.add_argument('slab_for_1', help='format: db_path.db@row_nr')
    parser.add_argument('neb_image_2', help='format: db_path.db@row_nr')
    parser.add_argument('slab_for_2', help='format: db_path.db@row_nr')
    args = parser.parse_args()

    main(**args.__dict__)