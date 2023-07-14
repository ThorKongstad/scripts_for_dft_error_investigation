import os
import time
from typing import NoReturn, Sequence, Tuple, Never, Optional
from dataclasses import dataclass, field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
import ase.db as db
import pandas as pd
from sqlite3 import OperationalError


@dataclass
class reaction:
    reactants: Sequence[Tuple[str, float]]
    products: Sequence[Tuple[str, float]]
    experimental_ref: float

    def toStr(self) -> str:
        return ' + '.join([f'{n:.2g}{smi if smi != "cid281" else "C|||O"}' for smi, n in self.reactants]) + ' ---> ' + ' + '.join([f'{n:.2g}{smi  if smi != "cid281" else "C|||O"}' for smi, n in self.products])


def build_pd(db_dir_list, select_key: Optional = None):
    if isinstance(db_dir_list, str): db_dir_list = [db_dir_list]
    db_list = [db.connect(work_db) for work_db in db_dir_list]
    pd_dat = pd.DataFrame([row.__dict__ for work_db in db_list for row in work_db.select(selection=select_key)])
    return pd_dat


def sanitize(unclean_str: str) -> str:
    for ch in ['!', '*', '?', '{', '[', '(', ')', ']', '}', "'", '"']: unclean_str = unclean_str.replace(ch, '')
    for ch in ['/', '\\', '|', ' ', ',', '.']: unclean_str = unclean_str.replace(ch, '_')
    for ch in ['=', '+', ':', ';']: unclean_str = unclean_str.replace(ch, '-')
    return unclean_str



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
        reaction((('[HH]',1),('O=O',0.5)),(('O',1),),None),
        reaction((('[HH]', 1), ('O=O', 1)), (('OO', 1),), None),
    ]

all_reactions = reactions + combustion_reactions