from csv import DictReader
from os import path
from typing import List, Tuple

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


def load_bp() -> Tuple[List[str], List[List[float]]]:
    """
    Loads boiling point data; target values given in Celsius

    Returns:
        Tuple[List[str], List[List[float]]]: (smiles strings, target values);
            target values have shape (n_samples, 1)
    """

    return _load_set('bp')


def load_cn() -> Tuple[List[str], List[List[float]]]:
    """
    Loads cetane number data; target values given in CN units

    Returns:
        Tuple[List[str], List[List[float]]]: (smiles strings, target values);
            target values have shape (n_samples, 1)
    """

    return _load_set('cn')


def load_cp() -> Tuple[List[str], List[List[float]]]:
    """
    Loads cloud point data; target values given in Celsius

    Returns:
        Tuple[List[str], List[List[float]]]: (smiles strings, target values);
            target values have shape (n_samples, 1)
    """

    return _load_set('cp')


def load_kv() -> Tuple[List[str], List[List[float]]]:
    """
    Loads kinematic viscosity data; target values given in mm^2/s = cSt

    Returns:
        Tuple[List[str], List[List[float]]]: (smiles strings, target values);
            target values have shape (n_samples, 1)
    """

    return _load_set('kv')


def load_lhv() -> Tuple[List[str], List[List[float]]]:
    """
    Loads lower heating value data; target values given in MJ/kg = kJ/g

    Returns:
        Tuple[List[str], List[List[float]]]: (smiles strings, target values);
            target values have shape (n_samples, 1)
    """

    return _load_set('lhv')


def load_mon() -> Tuple[List[str], List[List[float]]]:
    """
    Loads motor octane number data; target values given in MON units

    Returns:
        Tuple[List[str], List[List[float]]]: (smiles strings, target values);
            target values have shape (n_samples, 1)
    """

    return _load_set('mon')


def load_pp() -> Tuple[List[str], List[List[float]]]:
    """
    Loads pout point data; target values given in Celsius

    Returns:
        Tuple[List[str], List[List[float]]]: (smiles strings, target values);
            target values have shape (n_samples, 1)
    """

    return _load_set('pp')


def load_ron() -> Tuple[List[str], List[List[float]]]:
    """
    Loads research octane number data; target values given in RON units

    Returns:
        Tuple[List[str], List[List[float]]]: (smiles strings, target values);
            target values have shape (n_samples, 1)
    """

    return _load_set('ron')


def load_ysi() -> Tuple[List[str], List[List[float]]]:
    """
    Loads yield sooting index data; target values given in unified YSI units

    Returns:
        Tuple[List[str], List[List[float]]]: (smiles strings, target values);
            target values have shape (n_samples, 1)
    """

    return _load_set('ysi')
