import argparse
import os.path
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
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
#    dis_div: float
    step_row: AtomsRow
#    ref_image: AtomsRow
    dft_E: float = field(init=False)
    ensamble_Es: np.ndarray = field(init=False)
    ensamble_mean: float = field(init=False)
    ensamble_sd: float = field(init=False)
#    upper: float | int = field(init=False)
#    lower: float | int = field(init=False)

    def __post_init__(self):
        self.dft_E = self.step_row.get('energy') #-self.ref_image.get('energy')
        self.ensamble_Es = np.array(self.step_row.data.get('ensemble_en'))

        self.ensamble_mean = mean(self.ensamble_Es)
        self.ensamble_sd = sd(self.ensamble_Es, self.ensamble_mean)
#        self.upper = self.ensamble_mean + self.ensamble_sd
#        self.lower = self.ensamble_mean - self.ensamble_sd


def ends_with(string: str, end_str: str) -> str:
    return string + end_str * (end_str != string[-len(end_str):0])


@retry(retry=retry_if_exception_type(FileExistsError), stop=stop_after_attempt(5), wait=wait_fixed(2))
def folder_exist(folder_name: str, path: str = '.') -> None:
    if folder_name not in os.listdir(path): os.mkdir(ends_with(path, '/') + folder_name)


def dis_atoms(at_row_1: AtomsRow, at_row_2: AtomsRow) -> float:
    return np.sqrt(sum(
            sum((pos1[i] - pos2[i]) ** 2 for i in range(3))
            for pos1, pos2 in zip(
                at_row_1.toatoms().get_positions(),
                at_row_2.toatoms().get_positions()
                )
            ))


def sd(values: Sequence[float], mean_value: Optional[float] = None) -> float:
    if not mean_value: mean_value = mean(values)
    return np.sqrt((1 / len(values)) * sum(((x - mean_value) ** 2 for x in values)))
def mean(values: Sequence[float]) -> float: return sum(values) / len(values)



def reaction_plotly(reaction_steps: Sequence[reaction_step], plot_name: str):
    transition_index = reaction_steps.index(max(reaction_steps, key=lambda reac: reac.dft_E - reaction_steps[0].dft_E))
    fig = go.Figure()

    #image_dis = [dis_atoms(reac_step.step_row, reaction_steps[0].step_row) for reac_step in reaction_steps]

    distance_moved = 0
    image_dis = [(distance_moved := distance_moved + dis_atoms(reac_step.step_row, reaction_steps[i-1 if i != 0 else 0].step_row)) for i, reac_step in enumerate(reaction_steps)]

    fig.add_trace(go.Scatter(
        name='Stderr in relation to initial state',
        x=image_dis + image_dis[::-1],
        y=[(reac_step.dft_E - reaction_steps[0].dft_E) + sd(reac_step.ensamble_Es - reaction_steps[0].ensamble_Es) for reac_step in reaction_steps] + [(reac_step.dft_E - reaction_steps[0].dft_E) - sd(reac_step.ensamble_Es - reaction_steps[0].ensamble_Es) for reac_step in reversed(reaction_steps)],
        fill='toself',
        fillcolor='red',
        opacity=0.2,
        mode='none',
    ))

    fig.add_trace(go.Scatter(
        name='Stderr in relation to transition state',
        x=image_dis + image_dis[::-1],
        y=[(reac_step.dft_E - reaction_steps[0].dft_E) + sd(reac_step.ensamble_Es - reaction_steps[transition_index].ensamble_Es) for reac_step in reaction_steps] + [(reac_step.dft_E - reaction_steps[0].dft_E) - sd(reac_step.ensamble_Es - reaction_steps[transition_index].ensamble_Es) for reac_step in reversed(reaction_steps)],
        fill='toself',
        fillcolor='black',
        opacity=0.2,
        mode='none',
    ))

    fig.add_trace(go.Scatter(
        name='Stderr in relation to final state',
        x=image_dis + image_dis[::-1],
        y=[(reac_step.dft_E - reaction_steps[0].dft_E) + sd(reac_step.ensamble_Es - reaction_steps[-1].ensamble_Es) for reac_step in reaction_steps] + [(reac_step.dft_E - reaction_steps[0].dft_E) - sd(reac_step.ensamble_Es - reaction_steps[-1].ensamble_Es) for reac_step in reversed(reaction_steps)],
        fill='toself',
        fillcolor='blue',
        opacity=0.2,
        mode='none',
    ))

    fig.add_trace(go.Scatter(
        name='Neb reaction path',
        x=image_dis,
        y=[(reac_step.dft_E - reaction_steps[0].dft_E) for reac_step in reaction_steps],
        mode='lines',
        line=dict(
            color='black',
        ),
        hovertemplate=[r'DE from initial state:  %{y:.3} +- '+f'{sd(reac_step.ensamble_Es - reaction_steps[0].ensamble_Es):.3} eV<br>'
                       + f'DE from transition state:  {(reac_step.dft_E - reaction_steps[transition_index].dft_E):.3} +- {sd(reac_step.ensamble_Es - reaction_steps[transition_index].ensamble_Es):.3} eV<br>'
                       + f'DE from finale state:  {(reac_step.dft_E - reaction_steps[-1].dft_E):.3} +- {sd(reac_step.ensamble_Es - reaction_steps[-1].ensamble_Es):.3} eV<br>'
                    for reac_step in reaction_steps]
    ))

    fig.update_layout(
        xaxis_title='Distance between images',
        yaxis_title=r'$\Delta \text{E in relation to the initial state (eV)}$',

    )

    folder_exist('reaction_plots')
    save_name = f'reaction_plots/{plot_name}'
    fig.write_html(save_name + '.html', include_mathjax='cdn')


def main(image_db: str, plot_name: str):
    with db.connect(image_db) as neb_obj:
        neb_images = [reaction_step(i, row) for i, row in enumerate(neb_obj.select())]

    reaction_plotly(neb_images, plot_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument('image_db', type=str)
    parser.add_argument('save_name', type=str)
    args = parser.parse_args()

    main(image_db=args.image_db, plot_name=args.save_name)
