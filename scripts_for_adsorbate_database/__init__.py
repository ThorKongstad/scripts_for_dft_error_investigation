import os
import time
from typing import NoReturn
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
import ase.db as db
from sqlite3 import OperationalError

def sanitize(unclean_str: str) -> str:
    for ch in ['!', '*', '?', '{', '[', '(', ')', ']', '}', "'",'"']: unclean_str = unclean_str.replace(ch, '')
    for ch in ['/', '\\', '|',' ']: unclean_str = unclean_str.replace(ch, '_')
    for ch in ['=', '+', ':']: unclean_str = unclean_str.replace(ch, '-')
    return unclean_str


def folder_exist(folder_name: str, path: str = '.', tries: int = 10) -> NoReturn:
    try:
        tries -= 1
        if folder_name not in os.listdir(path): os.mkdir(ends_with(path, '/')+folder_name)
    except FileExistsError:
        time.sleep(2)
        if tries > 0: folder_exist(folder_name, path=path, tries=tries)

def ends_with(string: str, end_str: str) -> str:
    return string + end_str * (end_str != string[-len(end_str):0])


@retry(retry=retry_if_exception_type(OperationalError), stop=stop_after_attempt(5), wait=wait_fixed(10))
def update_db(db_dir: str, db_update_args: dict):
    with db.connect(db_dir) as db_obj:
        db_obj.update(**db_update_args)


