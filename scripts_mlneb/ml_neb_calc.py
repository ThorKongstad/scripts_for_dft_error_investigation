#partition=katla
#nprocshared=32
#mem=2000MB
#constrain='[v4|v5]'
#nodes=1

import argparse
from ase.io import read
from catlearn.optimize.mlneb import MLNEB
from gpaw import GPAW, PW
from typing import Optional


def main(path_file: str, restart_file: Optional[str] = None, out_prefix: Optional[str] = None):
#    path_atoms = read(path_file)

#    initial = path_atoms[0]
#    path = path_atoms[1:-1]
#    final = path_atoms[-1]

    initial = read(f'{path_file}', index='0')
    path = read(f'{path_file}', index='1:-1')
    final = read(f'{path_file}', index='-1')

    nimages = len(path)

    prefix = path_file.split('.')[0] if out_prefix is None else out_prefix

    restart_dic = {'restart': True, 'prev_calculations': restart_file} if restart_file is not None else {}

    neb_catlearn = MLNEB(start=initial,  # Initial end-point.
                         end=final,  # Final end-point.
                         ase_calc=GPAW(mode=PW(600),
                                       kpts=(4,4,1),
                                       basis='dzp',
                                       xc="RPBE",
                                       txt=f'{prefix}_ml-NEB.txt',
                                       parallel={'augment_grids':True,'sl_auto':True}),
                         # Calculator, it must be the same as the one used for the optimizations.
                         n_images=nimages,  # Number of images (interger or float, see above).
                         interpolation=path, #'idpp',
                         # Choose between linear or idpp interpolation (as implemented in ASE). Can also give it a path.traj file or list of atoms
                         name=prefix,  # prefix for the file names created by MLNEB
                         **restart_dic
                         )  # for restarting calculation. You can still change other parameters with this.
    neb_catlearn.run(fmax=0.03, trajectory=f'{prefix}_ml-NEB.traj', optimizer='MDMin', full_output=False, name=prefix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Directory to path file.')
    parser.add_argument('-res', '--restart_file', type=str, help='Directory to the restart file.')
    parser.add_argument('-n', '--name', type=str, help='prefix for the files created by ml neb')
    args = parser.parse_args()

    main(args.path, args.restart_file, args.name)

