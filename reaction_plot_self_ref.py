import argparse
import os.path
import ase.db as db
from ase.db.row import AtomsRow
import numpy as np
from dataclasses import dataclass, field
from typing import NoReturn, Sequence, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


@dataclass
class reaction_step:
    order: int
    rmsd_div: float
    step_row: AtomsRow
    ref_image: AtomsRow
    rpbe_E: float | int = field(init=False)
    ensamble_Es: Sequence[float | int] = field(init=False)
    ensamble_mean: float | int = field(init=False)
    ensamble_sd: float | int = field(init=False)
    upper: float | int = field(init=False)
    lower: float | int = field(init=False)

    def __post_init__(self):
        self.rpbe_E = self.step_row.get('energy')-self.ref_image.get('energy')
        self.ensamble_Es = [
            t_E-r_E
            for t_E, r_E in zip(
                self.step_row.data.get('ensemble_en'),
                self.ref_image.data.get('ensemble_en')
            )
        ]

        self.ensamble_mean = mean(self.ensamble_Es)
        self.ensamble_sd = sd(self.ensamble_Es, self.ensamble_mean)
        self.upper = self.ensamble_mean + self.ensamble_sd
        self.lower = self.ensamble_mean - self.ensamble_sd


def bind_E(total_E: float | int, slab_E: float | int, ad_E: float | int) -> float | int: return total_E - slab_E - ad_E


def sd(values: Sequence[int | float], mean_value: int | float) -> int | float: return np.sqrt(1 / len(values) * sum(((x - mean_value) ** 2 for x in values)))
def mean(values: Sequence[int | float]) -> float: return sum(values) / len(values)


def weighted_sd(dat: Sequence[float | int], coef: Sequence[float | int], mean: float | int) -> float | int:
    # https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf
    non_zero_coefs = len([co for co in coef if co != 0])
    return np.sqrt(
        sum(co_i * (dat_i - mean)**2 for co_i, dat_i in zip(coef, dat)) /
        (((non_zero_coefs-1)*sum(coef))/non_zero_coefs)
    )


def weighted_mean(dat: Sequence[float | int], coef: Sequence[float | int]) -> float | int: return sum(dat_i*coef_i for dat_i, coef_i in zip(dat, coef)) / sum(coef)


def rmsd_at_rows(at_row_1: AtomsRow, at_row_2: AtomsRow) -> float:
    return np.sqrt((1/at_row_1.natoms)*sum(
        ((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2 + (pos1[2]-pos2[2])**2)
        for pos1, pos2 in zip(
            at_row_1.toatoms().get_positions(),
            at_row_2.toatoms().get_positions()
        )
    ))


def reaction_plot_plotly(images: Sequence[Sequence[reaction_step]] | Sequence[reaction_step], name: Optional[str] = None):
    def violin_placement_picker(violin_nr_at_x: int) -> str: return 'positive' if violin_nr_at_x % 2 else 'negative'

    if isinstance(images[0], reaction_step): images = [images]
    plot = go.Figure()
    violin_placements = []
    for i, im_seq in enumerate(images):
        panda = pd.DataFrame(im_seq)

        plot.add_trace(go.Scatter(
            x=panda.get('rmsd_div').tolist() + panda.get('rmsd_div').tolist()[::-1],
            y=panda.get('upper').tolist() + panda.get('lower').tolist()[::-1],
            fill='toself',
            fillcolor='red',
            showlegend=False,
            opacity=0.2,
            mode='none',
            name='Standard deviation',
            hoverinfo='none'
        ))

        for index, row in panda.iterrows():
            violin_placements.append((cur_rmsd := row.get('rmsd_div')))
            ensemple_energies = row.get('ensamble_Es')
            plot.add_trace(go.Violin(
                y=ensemple_energies,
                x0=cur_rmsd,
                opacity=0.4,
                line_color='black',
                fillcolor='grey',
                showlegend=False,
                side=violin_placement_picker(violin_placements.count(cur_rmsd)) if len(images) > 1 else 'both',
            ))

        plot.add_trace(px.line(data_frame=panda,
                               x='rmsd_div',
                               y='rpbe_E',
                               markers=True,
                               error_y='ensamble_sd'
                               ).data[0])

    plot.update_layout(dragmode=False)

    plot.update_layout(legend_title_text=name)
    plot.update_xaxes(title_text="RMSD from the the first image")
    plot.update_yaxes(title_text="eV")
    plot.update_layout(hovermode='closest')

    if name: plot.write_html(name)
    else: plot.show()


def main(image_db: str, ref_image_nr: Sequence[int], show_bool: bool = False, name_tag: Optional[str] = None):

    if any(at in image_db.lower() for at in ['pd', 'pt']): neb_order = ['initial.traj', 'neb1.traj', 'neb2.traj', 'neb3.traj', 'neb4.traj', 'neb5.traj', 'final.traj']
    elif 'ag' in image_db.lower(): neb_order = ['Ag_top_fcc_IS.traj', 'neb1.traj', 'neb2.traj', 'neb3.traj', 'neb4.traj', 'neb5.traj', 'Ag_top_fcc_FS.traj']
    else: raise ValueError('could not find recognise the file and so could not determine order.')

    with db.connect(image_db) as neb_obj:
        structure_rows = [neb_obj.get(f'name={name}') for name in neb_order]
        step_seqs = [
            [reaction_step(i, rmsd_at_rows(stepRow, structure_rows[0]), stepRow, structure_rows[ref]) for i, stepRow in enumerate(structure_rows)]
            for ref in ref_image_nr
        ]

    name = os.path.basename(image_db).split('.')[0]+(name_tag if name_tag else '') + '_plot.html'
    reaction_plot_plotly(step_seqs, name if not show_bool else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_db', type=str)
    parser.add_argument('-ref', '--reference_image', nargs='+', default=[0], type=int, help='the index of the image that will be the reference, note that indexing starts from 0')
    parser.add_argument('-show', '--show', action='store_true')
    parser.add_argument('-tag', '--tag', help='tag to be added to the save file.')
    args = parser.parse_args()

    main(args.image_db, args.reference_image, args.show, name_tag=args.tag)
