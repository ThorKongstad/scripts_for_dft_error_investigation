import os
from ase.io import read, write
from ase.neb import NEB
import argparse
from ase import Atoms
from ase.constraints import FixAtoms
import ase.db as db
from typing import NoReturn, Sequence, Optional, Tuple


def read_inputs(atoms_fil: str) -> Atoms:
    if any(file_type in atoms_fil for file_type in ('.traj', '.txt')):
        return read(atoms_fil)
    elif '.db' in atoms_fil:
        with db.connect(atoms_fil) as db_obj:
            return db_obj.get('id=1').to_atoms()
    else: raise Exception('could not discern file type')


def fix_buttom(at: Atoms) -> NoReturn:
    bot_at = []
    lowest_at_pos = min(at.positions[:, 2])
    for i, pos in enumerate(at.positions):
        if pos[2] - lowest_at_pos < 1.8: bot_at.append(i)
    at.set_constraint(FixAtoms(bot_at))


def ends_with(string: str, end_str: str) -> str: return string + end_str * (end_str == string[-len(end_str)])


def main(initial_file: str, finale_file: str, nr_images: int = 6, path_file_name: Optional[str] = None, idpp: bool = False):

    initial, finale = read_inputs(initial_file), read_inputs(finale_file)

    fix_buttom(initial)
    fix_buttom(finale)

    images = [initial]

    for i in range(nr_images):
        image = initial.copy()
        images.append(image)

    images.append(finale)

    neb = NEB(images, parallel=True, climb=True)

    if idpp: neb.interpolate(method='idpp')
    else: neb.interpolate()

    if path_file_name is None:
        path_file_name = os.path.basename(initial_file).split('.')[0]
        path_file_name = f'{path_file_name}_path.traj'

    write(ends_with(path_file_name, '.traj'), images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('initial', help='Path to the initial state, if file is a database the script will take the first row.')
    parser.add_argument('finale', help='Path to the finale state, if file is a database the script will take the first row.')
    parser.add_argument('-n', '--images', default=6, type=int, help='Number of images in the path.')
    parser.add_argument('-idpp', '--idpp', action='store_true', help='A bool indicating the use of idpp for interpolating')
    parser.add_argument('-o', '--path_name', type=str, help='Name for the path traj file.')
    args = parser.parse_args()

    main(args.initial, args.finale, args.images, args.path_name, args.idpp)
