#partition=katla_verylong
#nprocshared=32
#mem=2000MB
#constrain='[v1|v2|v3|v4|v5]'

from gpaw import GPAW, PW, Davidson
from gpaw.utilities import h2gpts
from ase.collections import g2
from ase import Atoms
from dataclasses import dataclass
from typing import Sequence, NoReturn, Tuple
from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo
from ase.parallel import parprint, world



@dataclass
class reaction:
    reactants: Sequence[Tuple[str,int|float]]
    products: Sequence[Tuple[str,int|float]]
    experimental_ref: float

    def toStr(self) -> str:
        return ' + '.join([f'{n}{smi if smi != "cid281" else "C|||O"}' for smi,n in self.reactants]) + ' ---> ' + ' + '.join([f'{n}{smi  if smi != "cid281" else "C|||O"}' for smi,n in self.products])


def main():
    example_reaction = reaction((('C1=CC=CC=C1', 1/6), ('O=O', 7/6)), (('C(=O)=O', 1), ('O', 0.5)), -33.83941093/6)  # 10

    benzene_PBE: Atoms = g2['C6H6']
    benzene_PBE.set_cell([10,10,10])
    benzene_PBE.center()
    calc = GPAW(mode='fd',
                xc='PBE',
                basis='dzp',
                txt=f'benzene_PBE.txt',
                gpts=h2gpts(0.16, benzene_PBE.get_cell(), idiv=4),
                parallel={'augment_grids': True, 'sl_auto': True},
                convergence={'eigenstates': 0.000001},
                eigensolver=Davidson(3),
                symmetry='off',
                )

    benzene_PBE.set_calculator(calc)
    benzene_PBE.get_potential_energy()
    benzene_PBE_vib = Vibrations(benzene_PBE, name=f'benzene_PBE_vib.txt')
    benzene_PBE_vib.run()
    benzene_PBE_thermo = IdealGasThermo(
        vib_energies=(val for val in benzene_PBE_vib.get_energies() if val.isreal()), # will likely fail because of imaginary vals fix
        geometry='nonlinear',
        potentialenergy=benzene_PBE.get_potential_energy(),
        atoms=benzene_PBE,
    )


    oxygen_PBE: Atoms = g2['O2']
    oxygen_PBE.set_cell([10,10,10])
    oxygen_PBE.center()
    calc = GPAW(mode='fd',
                xc='PBE',
                basis='dzp',
                txt=f'oxygen_PBE.txt',
                gpts=h2gpts(0.16, oxygen_PBE.get_cell(), idiv=4),
                parallel={'augment_grids': True, 'sl_auto': True},
                convergence={'eigenstates': 0.000001},
                eigensolver=Davidson(3),
                symmetry='off',
                hund=True
                )

    oxygen_PBE.set_calculator(calc)
    oxygen_PBE.get_potential_energy()
    oxygen_PBE_vib = Vibrations(oxygen_PBE, name=f'oxygen_PBE_vib.txt')
    oxygen_PBE_vib.run()
    oxygen_PBE_thermo = IdealGasThermo(
        vib_energies=oxygen_PBE_vib.get_energies(),
        geometry='linear',
        potentialenergy=oxygen_PBE.get_potential_energy(),
        atoms=oxygen_PBE,
        spin=1
    )


    carbon_dioxide_PBE: Atoms = g2['CO2']
    carbon_dioxide_PBE.set_cell([10,10,10])
    carbon_dioxide_PBE.center()
    calc = GPAW(mode='fd',
                xc='PBE',
                basis='dzp',
                txt=f'carbon_dioxide_PBE.txt',
                gpts=h2gpts(0.16, carbon_dioxide_PBE.get_cell(), idiv=4),
                parallel={'augment_grids': True, 'sl_auto': True},
                convergence={'eigenstates': 0.000001},
                eigensolver=Davidson(3),
                symmetry='off',
                )

    carbon_dioxide_PBE.set_calculator(calc)
    carbon_dioxide_PBE.get_potential_energy()
    carbon_dioxide_PBE_vib = Vibrations(carbon_dioxide_PBE, name=f'carbon_dioxide_PBE_vib.txt')
    carbon_dioxide_PBE_vib.run()
    carbon_dioxide_PBE_thermo = IdealGasThermo(
        vib_energies=carbon_dioxide_PBE_vib.get_energies(),
        geometry='linear',
        potentialenergy=carbon_dioxide_PBE.get_potential_energy(),
        atoms=carbon_dioxide_PBE,
    )


    water_PBE: Atoms = g2['H2O']
    water_PBE.set_cell([10,10,10])
    water_PBE.center()
    calc = GPAW(mode='fd',
                xc='PBE',
                basis='dzp',
                txt=f'water_PBE.txt',
                gpts=h2gpts(0.16, water_PBE.get_cell(), idiv=4),
                parallel={'augment_grids': True, 'sl_auto': True},
                convergence={'eigenstates': 0.000001},
                eigensolver=Davidson(3),
                symmetry='off',
                )

    water_PBE.set_calculator(calc)
    water_PBE.get_potential_energy()
    water_PBE_vib = Vibrations(water_PBE, name=f'water_PBE_vib.txt')
    water_PBE_vib.run()
    carbon_dioxide_PBE_vib.run()
    water_PBE_thermo = IdealGasThermo(
        vib_energies=water_PBE_vib.get_energies(),
        geometry='nonlinear',
        potentialenergy=water_PBE.get_potential_energy(),
        atoms=water_PBE,
    )

    reac_enthalpy = benzene_PBE_thermo.get_enthalpy(temperature=298.15,verbose=False)*example_reaction.reactants[0][1] + oxygen_PBE_thermo.get_enthalpy(temperature=298.15,verbose=False)*example_reaction.reactants[1][1]
    prod_enthalpy = carbon_dioxide_PBE_thermo.get_enthalpy(temperature=298.15,verbose=False)*example_reaction.products[0][1] + water_PBE_thermo.get_enthalpy(temperature=298.15,verbose=False)*example_reaction.products[1][1]

    reaction_enthalpy = reac_enthalpy - prod_enthalpy

    parprint("reaction test for PBE-vdW")
    parprint(f"finale reaction enthalpy: {reaction_enthalpy}")
    parprint("")
    parprint(f"benzene dft energy: {benzene_PBE.get_potential_energy()}")
    parprint("benzene vib")
    parprint(benzene_PBE_vib.summary())
    parprint("")
    parprint("benzene thermodynamic")
    if world == 0: benzene_PBE_thermo.get_enthalpy(temperature=298.15)
    parprint("")
    parprint("")

    parprint(f"oxygen dft energy: {oxygen_PBE.get_potential_energy()}")
    parprint("oxygen vib")
    parprint(oxygen_PBE_vib.summary())
    parprint("")
    parprint("oxygen thermodynamic")
    if world == 0: oxygen_PBE_thermo.get_enthalpy(temperature=298.15)
    parprint("")
    parprint("")

    parprint(f"carbon dioxide dft energy: {carbon_dioxide_PBE.get_potential_energy()}")
    parprint("carbon dioxide vib")
    parprint(carbon_dioxide_PBE_vib.summary())
    parprint("")
    parprint("carbon dioxide thermodynamic")
    if world == 0: carbon_dioxide_PBE_thermo.get_enthalpy(temperature=298.15)
    parprint("")
    parprint("")

    parprint(f"water dft energy: {water_PBE.get_potential_energy()}")
    parprint("water vib")
    parprint(water_PBE_vib.summary())
    parprint("")
    parprint("water thermodynamic")
    if world == 0: water_PBE_thermo.get_enthalpy(temperature=298.15)
    parprint("")
    parprint("")

if __name__ == '__main__':
    main()