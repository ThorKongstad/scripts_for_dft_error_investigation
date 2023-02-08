import argparse
import ase.db as db
import re

def main(fil):
    with open(fil, 'r') as vib_e_fil:
        vib_e_str = vib_e_fil.read()

    with db.connect('/groups/kemi/thorkong/errors_investigation/molreact.db') as db_obj:
        row_id = int(re.match(r'(?:.*/)?(?:vib_en_.*_)(?P<id>\d*)(?:\.txt)',fil).group('id'))
        db_obj.update(row_id, vibration=True, vib_en=vib_e_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vib_en_fil')
    args = parser.parse_args()

    main(args.vib_en_fil)