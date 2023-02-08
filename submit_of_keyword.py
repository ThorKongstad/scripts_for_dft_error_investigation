import argparse
import ase.db as db
from ase.db.row import AtomsRow
import os
from typing import Sequence, Callable

def isgga(row: AtomsRow) -> bool: return row.get('xc') in ('PBE','RPBE','BEEF-vdW',"{'name':'BEEF-vdW','backend':'libvdwxc'}")
def ismgga(row: AtomsRow) -> bool: return row.get('xc') in ('MGGA_X_REVM06_L+MGGA_C_REVM06_L','MGGA_X_TPSS+MGGA_C_TPSS','MGGA_X_R2SCAN+MGGA_C_R2SCAN','MGGA_X_R4SCAN+MGGA_C_R2SCAN')
def isbee(row: AtomsRow) -> bool: return row.get('xc') in ('BEEF-vdW',"{'name':'BEEF-vdW','backend':'libvdwxc'}")
def coll_not_exist(row: AtomsRow, coll) -> bool: return row.get(coll) is None

def multi_filter_or(row:AtomsRow,funcs:Sequence[Callable[[AtomsRow],bool]]) -> bool: return any(func(row) for func in funcs)
def multi_filter_and(row:AtomsRow,funcs:Sequence[Callable[[AtomsRow],bool]]) -> bool: return all(func(row) for func in funcs)


def main(key: str, python_scibt: str, filter:str|None = None):
    func_list = []
    if filter is not None:
        for fil in filter.split(','):
            temp_func_list = []
            for fi in fil.split('&&'):
                if fi == 'isgga': temp_func_list += [isgga]
                elif fi == 'ismgga': temp_func_list += [ismgga]
                elif fi == 'isbee': temp_func_list += [isbee]
                elif fi.split('=')[0] == 'coll_not_exist' or fil.split('=')[0].lower() == 'collnotexist': temp_func_list += [lambda x: coll_not_exist(x,fil.split('=')[-1])]
                else: print(f'{fil} for was not recognised as implemented filter')
            if len(temp_func_list) != 0: func_list += [lambda x: multi_filter_and(x,temp_func_list.copy())]
    if len(func_list) != 0: filter = {'filter': lambda x: multi_filter_or(x,func_list)}
    else: filter={}


    with db.connect('/groups/kemi/thorkong/errors_investigation/molreact.db') as db_obj:
        row_iter = db_obj.select(selection=key,**filter)

    for row in row_iter:
        os.system(f'/groups/kemi/thorkong/errors_investigation/submit_katla_omp {python_scibt} {row.get("id")}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('keyword', help='must match keywords used in db selection')
    parser.add_argument('python_scipt')
    parser.add_argument('--filter','-f', help='current implemented filters are isgga, ismgga and collNotExist="COLLOM" t. a "," denotes an or and "&&" denotes an and')
    args = parser.parse_args()

    main(args.keyword, args.python_scipt,args.filter)
