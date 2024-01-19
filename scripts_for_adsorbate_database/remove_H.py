import argparse
from copy import deepcopy

from ase.io import read, write
from ase import Atoms


def main(original_struc, out_trajectory_name):
    old_atoms: Atoms = read(original_struc)
    constraints = deepcopy(old_atoms.constraints)
    old_atoms.set_constraint()

    del old_atoms[[at.index for at in old_atoms if at.symbol == 'H']]

    old_atoms.set_constraint(constraints)

    write(out_trajectory_name, old_atoms)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('struc_source')
    parser.add_argument('out_traj')
    args = parser.parse_args()

    main(args.struc_source, args.out_traj)
