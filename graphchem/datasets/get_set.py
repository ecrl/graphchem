from csv import DictReader
from os import path

_CSV_PATH = path.join(
    path.dirname(path.abspath(__file__)),
    'static'
)


def _load_set(prop: str) -> tuple:

    filename = path.join(_CSV_PATH, '{}.csv'.format(prop))
    with open(filename, 'r') as csv_file:
        reader = DictReader(csv_file)
        rows = [r for r in reader]
    csv_file.close()
    return (
        [r['SMILES'] for r in rows],
        [[float(r['{}'.format(prop).upper()])] for r in rows]
    )


def load_bp() -> tuple:

    return _load_set('bp')


def load_cn() -> tuple:

    return _load_set('cn')


def load_cp() -> tuple:

    return _load_set('cp')


def load_kv() -> tuple:

    return _load_set('kv')


def load_lhv() -> tuple:

    return _load_set('lhv')


def load_mon() -> tuple:

    return _load_set('mon')


def load_pp() -> tuple:

    return _load_set('pp')


def load_ron() -> tuple:

    return _load_set('ron')


def load_ysi() -> tuple:

    return _load_set('ysi')
