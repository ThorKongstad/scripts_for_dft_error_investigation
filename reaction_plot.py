import argparse
import os.path
import ase.db as db
from ase.db.row import AtomsRow
import numpy as np
from dataclasses import dataclass, field
from typing import NoReturn, Sequence, Optional, Tuple
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

@dataclass
class reaction_step:
    order: int
    rmsd_div: float
    step_row: AtomsRow
    slab_row: AtomsRow
    adsorbate_row: AtomsRow | Sequence[AtomsRow]
    rpbe_E: float | int = field(init=False)
    ensamble_Es: Sequence[float | int] = field(init=False)
    ensamble_mean: float | int = field(init=False)
    ensamble_sd: float | int = field(init=False)
    upper: float | int = field(init=False)
    lower: float | int = field(init=False)

    def __post_init__(self):
        def get_add_energy(ad_rows: AtomsRow | Sequence[AtomsRow]) -> Tuple[float, Sequence[float]]:
            ad_rows = (ad_rows,) if isinstance(ad_rows,AtomsRow) else ad_rows
            names = [row.get('name') for row in ad_rows]
            if all('C(=O)=O' in name for name in names):
                return ad_rows[0].get('energy'), ad_rows[0].data.get('ensemble_en')
            elif all(('o=o' in name.lower() or 'cid281' in name.lower()) for name in names):
                for row in ad_rows:
                    if 'o=o' in row.get('name').lower(): O2_row = row
                    elif 'cid281' in row.get('name').lower(): CO_row = row
                assert 'O2_row' in locals()
                assert 'CO_row' in locals()
                return CO_row.get('energy')+O2_row.get('energy')/2, [co_en+o2_en/2 for co_en,o2_en in zip(CO_row.data.get('ensemble_en'),O2_row.data.get('ensemble_en'))]
            elif len(ad_rows) == 1: return ad_rows[0].get('energy'), ad_rows[0].data.get('ensemble_en')
            else: raise Exception('get_add_energy haven\'t been implemented for current variation of solvent')

        ad_E, ad_ensamble_E = get_add_energy(self.adsorbate_row)

        self.rpbe_E = bind_E(self.step_row.get('energy'), self.slab_row.get('energy'),ad_E)
        self.ensamble_Es = [
            bind_E(t_E, s_E, a_E)
            for t_E, s_E, a_E in zip(
                self.step_row.data.get('ensemble_en'),
                self.slab_row.data.get('ensemble_en'),
                ad_ensamble_E
            )
        ]

        self.ensamble_mean = mean(self.ensamble_Es)
        self.ensamble_sd = sd(self.ensamble_Es,self.ensamble_mean)
        self.upper = self.ensamble_mean + self.ensamble_sd
        self.lower = self.ensamble_mean - self.ensamble_sd


def bind_E(total_E: float | int, slab_E: float | int, ad_E: float | int) -> float | int: return total_E - slab_E - ad_E


def weighted_mean(dat: Sequence[float|int], coef: Sequence[float|int]) -> float|int:
    return sum(dat_i*coef_i for dat_i, coef_i in zip(dat, coef)) / sum(coef)

def mean(values): return sum(values) / len(values)
def sd(values, mean_value): return np.sqrt(1 / len(values) * sum(((x - mean_value) ** 2 for x in values)))


def weighted_sd(dat: Sequence[float | int], coef: Sequence[float | int], mean: float | int) -> float | int:
    # https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf
    non_zero_coefs = len([co for co in coef if co != 0])
    return np.sqrt(
        sum(co_i * (dat_i - mean)**2 for co_i,dat_i in zip(coef, dat)) /
        (((non_zero_coefs-1)*sum(coef))/non_zero_coefs)
    )


def rmsd_at_rows(at_row_1: AtomsRow, at_row_2: AtomsRow):
    return np.sqrt((1/at_row_1.natoms)*sum(
        ((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2 + (pos1[2]-pos2[2])**2)
        for pos1,pos2 in zip(
            at_row_1.toatoms().get_positions(),
            at_row_2.toatoms().get_positions()
        )
    ))


def reaction_plot_plotly(images:Sequence[reaction_step], name: Optional[str] = None):
    panda = pd.DataFrame(images)

    plot = go.Figure()

    plot.add_trace(go.Scatter(
        x=panda.get('order').tolist() + panda.get('order').tolist()[::-1],
        y=panda.get('upper').tolist() + panda.get('lower').tolist()[::-1],
        fill='toself',
        fillcolor='red',
        showlegend=False,
        opacity=0.2,
        mode='none',
        name='Standard deviation'
    ))

    for index,row in panda.iterrows():
        ensemple_energies = row.get('ensamble_Es')
        plot.add_trace(go.Violin(
            y=ensemple_energies,
            x0=index, #list(row.get('order'))*len(ensemple_energies),
            opacity=0.4,
            line_color='black',
            fillcolor='grey',
            showlegend=False,
            #name='Bee ensemble distribution'
        ))

    plot.add_trace(px.line(data_frame =panda,
                   x='order',
                   y='rpbe_E',
                   markers=True,
                   error_y='ensamble_sd'
    ).data[0])

    plot.update_layout(legend_title_text=name)
    plot.update_xaxes(title_text="RMSD from the the first image")
    plot.update_yaxes(title_text="eV")

    if name: plot.write_html(name)
    else: plot.show()


def main(image_db: str, slab_db: str, ad_db: Sequence[str], show_bool: bool = False):
    if any(at in image_db.lower() for at in ['pd','pt']): neb_order = ['initial.traj', 'neb1.traj', 'neb2.traj', 'neb3.traj', 'neb4.traj', 'neb5.traj', 'final.traj']
    elif 'ag' in image_db.lower(): neb_order = ['Ag_top_fcc_IS.traj', 'neb1.traj', 'neb2.traj', 'neb3.traj', 'neb4.traj', 'neb5.traj', 'Ag_top_fcc_FS.traj']
    else: raise ValueError('could not find recognise the file and so could not determine order.')

    if len(ad_db) == 1:
        ad_db = ad_db[0]
        with db.connect(image_db) as neb_obj, db.connect(slab_db) as slab_obj, db.connect(ad_db) as ad_obj:
            structure_rows = [neb_obj.get(f'name={name}') for name in neb_order]
            step_seq = [reaction_step(i, rmsd_at_rows(stepRow, structure_rows[0]), stepRow, slab_obj.get(1), ad_obj.get(1)) for i, stepRow in enumerate(structure_rows)]
    elif len(ad_db) == 2:
        with db.connect(image_db) as neb_obj, db.connect(slab_db) as slab_obj, db.connect(ad_db[0]) as ad_1_obj,db.connect(ad_db[1]) as ad_2_obj:
            structure_rows = [neb_obj.get(f'name={name}') for name in neb_order]
            step_seq = [reaction_step(i, rmsd_at_rows(stepRow, structure_rows[0]), stepRow, slab_obj.get(1), (ad_1_obj.get(1),ad_2_obj.get(1))) for i, stepRow in enumerate(structure_rows)]

    name = os.path.basename(image_db).split('.')[0]+'_plot.html'
    reaction_plot_plotly(step_seq, name if not show_bool else None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_db', type=str, help='path to the neb image database, a reading order currently have to be modified for the file in main')
    parser.add_argument('slab_db', type=str,help='path to the slab database, will only read the first line.')
    parser.add_argument('adsorbate_db', type=str, nargs='+', help='path to the slab database, specific method for calculating the adsorbate energy ')
    parser.add_argument('-show', '--show', action='store_true', help='a bool to denote whether the script should show the html object or save it.')
    args = parser.parse_args()

    main(args.image_db, args.slab_db, args.adsorbate_db, args.show)
