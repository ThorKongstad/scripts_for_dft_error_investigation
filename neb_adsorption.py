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


def main(neb_image: str, slab: str, adsorbate: str):

    neb_adr = neb_image.split('@')
    slab_adr = slab.split('@')
    ads_adr = adsorbate.split('@')

    neb_pd = build_pd(neb_adr[0], select_key=(f'id={neb_adr[1]}' if len(neb_adr) > 1 else None))
    slab_pd = build_pd(slab_adr[0], select_key=(f'id={slab_adr[1]}' if len(slab_adr) > 1 else None))
    ads_pd = build_pd(ads_adr[0], select_key=(f'id={ads_adr[1]}' if len(ads_adr) > 1 else None))

    energy_diff = neb_pd.iloc[0].loc['energy'] - slab_pd.iloc[0].loc['energy'] - ads_pd.iloc[0].loc['energy']

    neb_ens = bytes_to_object(neb_pd.iloc[0].loc['_data']).get('ensemble_en')
    slab_ens = bytes_to_object(slab_pd.iloc[0].loc['_data']).get('ensemble_en')
    ads_ens = bytes_to_object(ads_pd.iloc[0].loc['_data']).get('ensemble_en')

    energy_ensemble = [n_i - s_i - ad_i for n_i, s_i, ad_i in zip(neb_ens, slab_ens, ads_ens)]

    print(f'the adsorption energy is: {energy_diff:.3f} +- {sd(energy_ensemble, mean_value=mean(energy_ensemble)):.3f} eV')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('neb_image', help='reference format: db_path.db@row_nr')
    parser.add_argument('slab', help='reference format: db_path.db@row_nr')
    parser.add_argument('adsorbate', help='reference format: db_path.db@row_nr')
    args = parser.parse_args()

    main(**args.__dict__)