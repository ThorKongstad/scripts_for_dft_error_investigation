#partition=katla_day
#nprocshared=120
#mem=2300MB
#constrain='v1|v2|v3'
#nodes=1-6

from ase.io import read
from ase.constraints import FixAtoms
from ase.neb import NEB
from ase.optimize import FIRE
from gpaw.mpi import rank, size
from gpaw import GPAW, PW
from typing import Optional
import argparse


def main(path_file: str, restart_file: Optional[str] = None, out_prefix: Optional[str] = None):
    initial = read(f'{path_file}', index='0')
    path = read(f'{path_file}', index='1:-1')
    final = read(f'{path_file}', index='-1')

    nimages = len(path)
    n = size // nimages  # number of cpu's per image
    assert nimages * n == size

    prefix = path_file.split('.')[0] if out_prefix is None else out_prefix

    images=[initial]

    for i in range(nimages):
        ranks = range(i * n, (i + 1) * n)
        image = path[i]
        if rank in ranks:
             calc = GPAW(mode=PW(600),
                        kpts = (4,4,1),
                        basis='dzp',
                        xc = "RPBE",
                        communicator=ranks,
                        txt=f'{prefix}_ase-NEB.txt',
                        parallel = {'augment_grids':True,'sl_auto':True})
             image.set_calculator(calc)
        images.append(image)
    images.append(final)

    neb = NEB(images, parallel=True, climb=True)

    qn = FIRE(neb, logfile=f'{prefix}_fire_neb.log', trajectory=f'{prefix}_fire_neb.traj')
    qn.run(fmax=0.03)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Directory to path file.')
    #parser.add_argument('-res', '--restart_file', type=str, help='Directory to the restart file.')
    parser.add_argument('-n', '--name', type=str, help='prefix for the files created by ml neb')
    args = parser.parse_args()

    main(path_file=args.path, out_prefix=args.name)