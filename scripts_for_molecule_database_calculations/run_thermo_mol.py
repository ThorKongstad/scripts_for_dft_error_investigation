#partition=katla_short
#nprocshared=1
#mem=2000MB
#constrain='[v1|v2|v3|v4|v5]'

import argparse
import os
import ase.db as db
from ase.thermochemistry import IdealGasThermo
from ase.parallel import parprint, world
from dataclasses import dataclass
import re
from typing import NoReturn, Sequence, Tuple
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from sqlite3 import OperationalError


@dataclass
class mode:
    nr: int
    energy: float
    reci_length: float


def load_vib_en_str(st:str) -> Tuple[float,Sequence[mode]]:
    pattern = r'(?:-{21}\s.+\s-{21}\s)(?P<freq_dat>(\s+\d+(\s+\d+(\.\d+i?)?){2})+)(?:\s+-{21}\s+)(?P<zpe>(?:Zero-point energy:\s*)\d+(\.\d+)?)'
    vib_match = re.match(pattern, st)
    mode_list = [mode(*[float(nr) for nr in mo.split(' ') if len(nr)>0]) for mo in vib_match.group('freq_dat').split('\n') if 'i' not in mo]
    return float(vib_match.group('zpe').split(' ')[-1]), mode_list


@retry(retry=retry_if_exception_type(OperationalError), stop=stop_after_attempt(5), wait=wait_fixed(10))
def update_db(db_dir: str, db_update_args: dict):
    with db.connect(db_dir) as db_obj:
        db_obj.update(**db_update_args)


def main(db_id: int, db_dir: str = 'molreact.db'):
    if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir))>0 else '.'): raise FileNotFoundError("Can't find database")
    with db.connect(db_dir) as db_obj:
        row = db_obj.get(selection=f'id={db_id}')
        if not row.get('relaxed'): raise Exception(f"atoms at row id: {db_id} haven't been relaxed.")
        if not row.get('vibration'): raise Exception(f"atoms at row id: {db_id} haven't been vibrated.")
        atoms = row.toatoms()
        smile = row.get('smiles')
        functional = row.get('xc')
        #grid_spacing= row.get('grid_spacing')
        vib_en_str = row.get('vib_en')
        total_energy = row.get('energy')

    parprint(f'outstd of thermo calculation for db entry {db_id} with structure: {smile} and functional: {functional}')

    zpe, mode_list = load_vib_en_str(vib_en_str)
    geom = 'linear' if smile in ['[HH]','C(=O)=O','O=O','cid281'] else 'nonlinear'

    thermo_obj = IdealGasThermo(
        vib_energies=[mo.energy / 1000 for mo in mode_list],
        geometry=geom,
        potentialenergy=total_energy,
        atoms=atoms,
        spin=(0 if smile != 'O=O' else 1)
    )

    if world.rank == 0:
        update_db(db_dir, db_update_args=dict(id=db_id, enthalpy=thermo_obj.get_enthalpy(temperature=298.15), zpe=zpe))
        #with db.connect(db_dir) as db_obj:
        #    db_obj.update(db_id, enthalpy=thermo_obj.get_enthalpy(temperature=298.15),zpe=zpe)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_base_id', type=int)
    parser.add_argument('-db','--database',help='directory to the database, if not stated will look for molreact.db in pwd.', default='molreact.db')
    args = parser.parse_args()

    main(args.data_base_id, args.database)
