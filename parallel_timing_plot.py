import math
import re
import matplotlib.pyplot as plt
import argparse
from dataclasses import dataclass, field
from typing import ClassVar, Sequence, Any, Callable
from itertools import groupby
import plotly.graph_objects as go


@dataclass
class Call:
    name: str
    level: int
    # rank_nr: int
    T1: float  # absolute start time
    T2: float = None  # absolute end time
    __alignment: float = 0
    __t1: float = None
    __t2: float = None

    @property
    def alignment(self):
        return self.__alignment

    @alignment.setter
    def alignment(self,value):
        self.__alignment = value
        #set t1 and t2
        self.__t1 = self.T1 - self.alignment
        if self.T2 is not None: self.__t2 = self.T2 - self.alignment

    @property
    def t1(self):
        if self.__t1 is None: self.__t1 = self.T1 - self.alignment
        return self.__t1

    @t1.setter
    def t1(self, value): print('dont try and change t1, the alignment setter will update it.')

    @property
    def t2(self):
        assert self.T2 is not None, 'T2 have to be set before a t2 can be defined.'
        if self.__t2 is None: self.__t2 = self.T2 - self.alignment
        return self.__t2

    @t2.setter
    def t2(self, value): print('dont try and change t2, the alignment setter will update it.')


@dataclass
class CallFunction:
    name: str
    number: int
    calls: list[Call] = field(default_factory=lambda: [])

    __thecolors: ClassVar[list] = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow',
                                   'darkred', 'indigo', 'springgreen', 'purple']
    __thehatches: ClassVar[list] = ['', '//', 'O', '*', 'o', r'\\', '.', '|']

    def __post_init__(self):
        self.color = self.__thecolors[self.number % len(self.__thecolors)]
        self.hatch = self.__thehatches[self.number // len(self.__thecolors)]


def load_timing_file(timing_file: str, interval: Sequence[int or float] | None = None) -> list[Call]:
    #the interval here deals in absolut time.
    pattern = r'T\d+ \S+ (?P<time>\S+) (?P<name>.+?) \(.*?\) (?P<action>started|stopped)'

    if isinstance(interval, float) or isinstance(interval, int): interval = (interval,)

    with open(timing_file, 'r') as work_file:
        file_content: str = work_file.read()

    matches = re.finditer(pattern, file_content)
    maxlevel = 0
    call_list = []
    level_tree = []
    for i, match in enumerate(matches):
        time, name, action = match.group('time', 'name', 'action')
        time = float(time)
        if interval is None or (interval[0] < time and (len(interval) == 1 or time > interval[1])):
            if action == 'started':
                level = len(level_tree)
                if level > maxlevel: maxlevel = level # max level is not saved nor used anywhere after this, in this function instance
                call = Call(name, level, time)
                level_tree.append(call)

            elif action == 'stopped':
                call = level_tree.pop()
                assert name == call.name
                call.T2 = time
                call_list.append(call)
        if interval is not None and len(interval) == 2 and time > interval[1]:
            # have to pack the last calls in the level tree of unfinished calls.
            for unfinished_call in reversed(level_tree):
                unfinished_call.T2 = time
                call_list.append(unfinished_call)
            break
    return call_list


def group_to_dict(ite: Sequence, key: Callable[[Any], float | int | str]) -> dict[Any:list[Any]]:
    iter_sorted = sorted(ite, key=key)
    dict_res = {key: list(val) for key, val in groupby(iter_sorted, key=key)}
    return dict_res


def ends_with(string: str, end_str: str) -> str:
    return string + end_str * (end_str != string[-len(end_str):0])


def plot_calls(call_list: list[list[Call]], interval: Sequence[int or float] = (0,), make_legend: bool = True, save_plot: str = 'core_timings.png'):
    if interval is None: interval = (0,)
    fig = plt.figure(figsize=(100, 100))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.08, right=.95, bottom=0.07, top=.95)

    def nested_list_generator(nes_list: list[list[any]]) -> any: # takes a nested list and yield each second lvl element
        for first_level in nes_list:
            for second_level in first_level: yield second_level

    functions = {}
    maxlevel = max(nested_list_generator(call_list),key=lambda x: x.level).level
    for rank_nr,rank in enumerate(call_list):
        print(f'starting plotting rank {rank_nr}')
        grouped_by_funct = group_to_dict(rank, lambda x: x.name)
        for key in grouped_by_funct.keys():
            if key not in functions.keys(): functions.update({key: CallFunction(key, len(functions))})
            work_call = [call for call in grouped_by_funct[key] if (len(interval) == 1 and call.t2 >= interval[0]) or (len(interval) == 2 and interval[0] <= call.t2 and interval[1] >= call.t1)]
            bar = ax.bar([call.t1 for call in work_call],
                         height=[1.0]*len(work_call),
                         width=[call.T2 - call.T1 for call in work_call],
                         bottom=[call.level + rank_nr * (maxlevel + 1)
                                 for call in work_call],
                         color=[functions[key].color],
                         hatch=functions[key].hatch,
                         edgecolor=['black'],
                         align='edge',
                         label=key)

    ax.set_xlim(left=interval[0])
    if len(interval) == 2: ax.set_xlim(right=interval[1])
    fig.savefig(save_plot)#,backend='cairo')
    #plt.savefig(save_plot)
#    for child in bar.get_children(): pass

    # make legend
    if make_legend:
        function_number_sqr = int(math.sqrt(1+len(functions.keys())))
        label_fig = plt.figure(figsize=(3*function_number_sqr,(4.8/19)*function_number_sqr+0.5))
        label_ax = label_fig.add_subplot(111)

        for key in functions.keys():
            label_ax.bar([0],[0],[0],[0],color=[functions[key].color], hatch=functions[key].hatch, label=key)
        label_ax.legend(handlelength=2.5,
                  labelspacing=0.0,
                  fontsize='large',
                  ncol= function_number_sqr,#1+len(functions.keys()) // 32,
                  mode='expand',
                  frameon=True,
                  loc='best')
        label_fig.savefig(fname=f'{save_plot.split(".")[0]}_legend.png')
    # make interactivity
    # save or show


def plotly_calls(call_list: list[list[Call]], interval: Sequence[int or float] = (0,), make_legend: bool = True, save_plot: str = 'core_timings'):
    fig = go.Figure()

    def nested_list_generator(nes_list: list[list[any]]) -> any: # takes a nested list and yield each second lvl element
        for first_level in nes_list:
            for second_level in first_level: yield second_level

    functions = {}
    maxlevel = max(nested_list_generator(call_list), key=lambda x: x.level).level
    for rank_nr,rank in enumerate(call_list):
        print(f'starting plotting rank {rank_nr}')
        grouped_by_funct = group_to_dict(rank, lambda x: x.name)
        for key in grouped_by_funct.keys():
            if key not in functions.keys(): functions.update({key: CallFunction(key, len(functions))})
            work_call = [call for call in grouped_by_funct[key] if (len(interval) == 1 and call.t2 >= interval[0]) or (len(interval) == 2 and interval[0] <= call.t2 and interval[1] >= call.t1)]
            for call in work_call:
                fig.add_trace(go.Scatter(
                    x=[call.T1, call.T1, call.T2, call.T2],
                    y=[call.level + rank_nr * (maxlevel + 1), (call.level + rank_nr * (maxlevel + 1))+1, (call.level + rank_nr * (maxlevel + 1))+1, call.level + rank_nr * (maxlevel + 1)],
                    mode='line',
                    line=dict(color=functions[key].color),
                    name=functions[key].name,
                    legendgroup=functions[key].name,
                    fill='toself',
                    fillcolor=functions[key].color,
                    fillpattern=functions[key].hatch,
                ))
                #fig.add_shape(type='rect', xref="x", yref="y",
                #              x0=call.T1, y0=call.level + rank_nr * (maxlevel + 1),
                #              x1=call.T2, y1=(call.level + rank_nr * (maxlevel + 1))+1,
                #              line=dict(color=functions[key].color),
                #              fillcolor=functions[key].color,
                #              name=functions[key].name)

    fig.write_html(ends_with(save_plot, '.html'), include_mathjax='cdn')


def silent_print(work, print_also=None):
    print(work)
    if print_also: print(print_also)
    return work

def main(time_files: str, alignment: str = None, interval=None,plot_save=None):
    # transform interval to sequence format
    if interval == None: interval = (0,)
    elif isinstance(interval,str): interval = [float(st) for st in interval.split('-')]
    elif isinstance(interval,int) or isinstance(interval,float): interval = (interval,)

    print('starts imports')
    # loading data
    calls = [work_calls for tfile in time_files if len(work_calls := load_timing_file(tfile)) > 0]

    # write assertion that makes sure that we get data out

    print('sets alignment')
    # setting alignment
    min_time = min(calls, key=lambda x: x[0].T1)[0].T1
    if alignment is None: alignment = min_time
    elif isinstance(alignment, str):
        try: alignment = float(alignment)
        except ValueError:
            # assumes alignment string refer to a specific function call
            min_call_time = []
            for core_calls in calls:
                calls_of_interest = [call for call in core_calls if call.name == alignment]
                if len(calls_of_interest) == 0: continue
                min_rank = min(calls_of_interest, key=lambda x: x.rank_nr)
                calls_min_level = [call for call in calls_of_interest if call.level == min_rank]
                min_call_time.append(min(calls_min_level, key=lambda x: x.T1).T1)
            assert len(min_call_time) != 0, 'Could not understand the alignment, it might be because the function ' \
                                            'name does not exit in timing files.'
            alignment = min(min_call_time)

    for core_calls in calls:
        for call in core_calls:
            call.alignment = alignment

    print('starts plotting')
    plotly_calls(calls,interval=interval,save_plot=plot_save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files',nargs='+')
    parser.add_argument('--interval', '-in', default=None, help='expected format is start-end or only start')
    parser.add_argument('--save', '-o', type=str, help='filename and path for the saving the plot')
    args = parser.parse_args()

    main(time_files=args.files, interval=args.interval,plot_save=args.save)
