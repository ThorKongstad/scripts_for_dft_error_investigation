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

    benzene_beef: Atoms = g2['C6H6']

    calc = GPAW(mode='fd',
                xc='BEEF_vdW',
                basis='dzp',
                txt=f'benzene_beef.txt',
                gpts=h2gpts(0.16, benzene_beef.get_cell(), idiv=4),
                parallel={'augment_grids': True, 'sl_auto': True},
                convergence={'eigenstates': 0.000001},
                eigensolver=Davidson(3),
                symmetry='off',
                )

    benzene_beef.set_calculator(calc)
    benzene_beef.get_potential_energy()
    benzene_beef_vib = Vibrations(benzene_beef, name=f'benzene_beef_vib.txt')
    benzene_beef_vib.run()
    benzene_beef_thermo = IdealGasThermo(
        vib_energies=(val for val in benzene_beef_vib.get_energies() if val.isreal()), # will likely fail because of imaginary vals fix
        geometry='nonlinear',
        potentialenergy=benzene_beef.get_potential_energy(),
        atoms=benzene_beef,
    )


    oxygen_beef: Atoms = g2['O2']

    calc = GPAW(mode='fd',
                xc='BEEF_vdW',
                basis='dzp',
                txt=f'oxygen_beef.txt',
                gpts=h2gpts(0.16, oxygen_beef.get_cell(), idiv=4),
                parallel={'augment_grids': True, 'sl_auto': True},
                convergence={'eigenstates': 0.000001},
                eigensolver=Davidson(3),
                symmetry='off',
                hund=True
                )

    oxygen_beef.set_calculator(calc)
    oxygen_beef.get_potential_energy()
    oxygen_beef_vib = Vibrations(oxygen_beef, name=f'oxygen_beef_vib.txt')
    oxygen_beef_vib.run()
    oxygen_beef_thermo = IdealGasThermo(
        vib_energies=oxygen_beef_vib.get_energies(),
        geometry='linear',
        potentialenergy=oxygen_beef.get_potential_energy(),
        atoms=oxygen_beef,
        spin=1
    )


    carbon_dioxide_beef: Atoms = g2['CO2']

    calc = GPAW(mode='fd',
                xc='BEEF_vdW',
                basis='dzp',
                txt=f'carbon_dioxide_beef.txt',
                gpts=h2gpts(0.16, carbon_dioxide_beef.get_cell(), idiv=4),
                parallel={'augment_grids': True, 'sl_auto': True},
                convergence={'eigenstates': 0.000001},
                eigensolver=Davidson(3),
                symmetry='off',
                )

    carbon_dioxide_beef.set_calculator(calc)
    carbon_dioxide_beef.get_potential_energy()
    carbon_dioxide_beef_vib = Vibrations(carbon_dioxide_beef, name=f'carbon_dioxide_beef_vib.txt')
    carbon_dioxide_beef_vib.run()
    carbon_dioxide_beef_thermo = IdealGasThermo(
        vib_energies=carbon_dioxide_beef_vib.get_energies(),
        geometry='linear',
        potentialenergy=carbon_dioxide_beef.get_potential_energy(),
        atoms=carbon_dioxide_beef,
    )


    water_beef: Atoms = g2['H2O']

    calc = GPAW(mode='fd',
                xc='BEEF_vdW',
                basis='dzp',
                txt=f'water_beef.txt',
                gpts=h2gpts(0.16, water_beef.get_cell(), idiv=4),
                parallel={'augment_grids': True, 'sl_auto': True},
                convergence={'eigenstates': 0.000001},
                eigensolver=Davidson(3),
                symmetry='off',
                )

    water_beef.set_calculator(calc)
    water_beef.get_potential_energy()
    water_beef_vib = Vibrations(water_beef, name=f'water_beef_vib.txt')
    water_beef_vib.run()
    carbon_dioxide_beef_vib.run()
    water_beef_thermo = IdealGasThermo(
        vib_energies=water_beef_vib.get_energies(),
        geometry='nonlinear',
        potentialenergy=water_beef.get_potential_energy(),
        atoms=water_beef,
    )

    reac_enthalpy = benzene_beef_thermo.get_enthalpy(temperature=298.15,verbose=False)*example_reaction.reactants[0][1] + oxygen_beef_thermo.get_enthalpy(temperature=298.15,verbose=False)*example_reaction.reactants[1][1]
    prod_enthalpy = carbon_dioxide_beef_thermo.get_enthalpy(temperature=298.15,verbose=False)*example_reaction.products[0][1] + water_beef_thermo.get_enthalpy(temperature=298.15,verbose=False)*example_reaction.products[1][1]

    reaction_enthalpy = reac_enthalpy - prod_enthalpy

    parprint("reaction test for BEEF-vdW")
    parprint(f"finale reaction enthalpy: {reaction_enthalpy}")
    parprint("")
    parprint(f"benzene dft energy: {benzene_beef.get_potential_energy()}")
    parprint("benzene vib")
    parprint(benzene_beef_vib.summary())
    parprint("")
    parprint("benzene thermodynamic")
    if world == 0: benzene_beef_thermo.get_enthalpy(temperature=298.15)
    parprint("")
    parprint("")

    parprint(f"oxygen dft energy: {oxygen_beef.get_potential_energy()}")
    parprint("oxygen vib")
    parprint(oxygen_beef_vib.summary())
    parprint("")
    parprint("oxygen thermodynamic")
    if world == 0: oxygen_beef_thermo.get_enthalpy(temperature=298.15)
    parprint("")
    parprint("")

    parprint(f"carbon dioxide dft energy: {carbon_dioxide_beef.get_potential_energy()}")
    parprint("carbon dioxide vib")
    parprint(carbon_dioxide_beef_vib.summary())
    parprint("")
    parprint("carbon dioxide thermodynamic")
    if world == 0: carbon_dioxide_beef_thermo.get_enthalpy(temperature=298.15)
    parprint("")
    parprint("")

    parprint(f"water dft energy: {water_beef.get_potential_energy()}")
    parprint("water vib")
    parprint(water_beef_vib.summary())
    parprint("")
    parprint("water thermodynamic")
    if world == 0: water_beef_thermo.get_enthalpy(temperature=298.15)
    parprint("")
    parprint("")

if __name__ == '__main__':
    main()