import argparse
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from scripts_for_adsorbate_database import update_db
from ase.io import read
from ase import Atoms


def main(db_index: int, txt_dir: str, db_dir: str, relaxed: bool = False):
    updated_atoms: Atoms = read(txt_dir, index=-1)
    update_db(db_dir,
              dict(
                  id=db_index,
                  atoms=updated_atoms,
                  relaxed=relaxed,
                  vibration=False,
                  vib_en=False
              ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db_index', type=int)
    parser.add_argument('txt_directory', help='Path to the gpaw output.')
    parser.add_argument('db_directory', help='Path to the database')
    parser.add_argument('-relaxed', '--relaxed', action='store_true', default=False)
    args = parser.parse_args()

    main(args.db_index, args.txt_directory, args.db_diretory, args.relaxed)
