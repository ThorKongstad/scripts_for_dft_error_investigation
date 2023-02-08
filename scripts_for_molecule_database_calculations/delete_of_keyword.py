import argparse
import ase.db as db
from typing import NoReturn

def main(key:str):

    raise 'not implemented'
    with db.connect('/groups/kemi/thorkong/errors_investigation/molreact.db') as db_obj:
        row_iter = db_obj.select(key)
        list_ids = []
        set_func = {}
        for row in row_iter:
            list_ids.append(row.get('id'))
            set_func.update('xc')

    print(f'Are you sure you want to delete objects with the key: {key}')
    print(f'This keyword fits these ids: {list_ids}')
    print(f'And among them are there these functionals: {set_func}')
    confirmation = input('Enter case sensitive YES or NO')

    if confirmation == 'YES':
        for id in reversed(sorted(list_ids)):
            with db.connect('/groups/kemi/thorkong/errors_investigation/molreact.db') as db_obj:
                db_obj.delete(id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('keyword', help='must match keywords used in db selection')
    args = parser.parse_args()

    main(args.keyword)