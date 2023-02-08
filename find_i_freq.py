import ase.db as db
import re


def main():
    with db.connect('/groups/kemi/thorkong/errors_investigation/molreact.db') as db_obj:
        dbo_sel = db_obj.select('vibration=True')

    for row in dbo_sel:
        if (ma := re.search('\d+(\.\d+)?i', row.get('vib_en'))) != None: print(f'id: {row.get("id")} lowest freq: {ma.group(0)}')
if __name__ == '__main__':
    main()

# {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,61,64,67,70,73,76,79,82,85,88,91,94,97,100,103,136,137,138,139}