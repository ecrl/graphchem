from csv import DictReader
from os import path
from typing import List, Tuple

import torch


_CSV_PATH = path.join(path.dirname(path.abspath(__file__)), 'static')


def _load_set(prop: str) -> Tuple[List[str], 'torch.tensor']:
    """ loads data for a given property

    Args:
        prop (str): property to obtain data for

    Returns:
        Tuple[List[str], torch.tensor]: (SMILES, property values)
    """

    filename = path.join(_CSV_PATH, f'{prop}.csv')
    with open(filename, 'r') as csv_file:
        reader = DictReader(csv_file)
        rows = [r for r in reader]
    csv_file.close()
    return (
        [r['SMILES'] for r in rows],
        torch.tensor([[float(r[f'{prop.upper()}'])] for r in rows]).type(
            torch.float32
        )
    )


def load_cn() -> Tuple[List[str], List[List[float]]]:
    """ loads cetane number data

    Returns:
        Tuple[List[str], List[List[float]]]: (SMILES, CN values)
    """

    return _load_set('cn')


def load_lhv() -> Tuple[List[str], List[List[float]]]:
    """ loads lower heating value data

    Returns:
        Tuple[List[str], List[List[float]]]: (SMILES, LHV values)
    """

    return _load_set('lhv')


def load_mon() -> Tuple[List[str], List[List[float]]]:
    """ loads motor octane number data

    Returns:
        Tuple[List[str], List[List[float]]]: (SMILES, MON values)
    """

    return _load_set('mon')


def load_ron() -> Tuple[List[str], List[List[float]]]:
    """ loads research octane number data

    Returns:
        Tuple[List[str], List[List[float]]]: (SMILES, RON values)
    """

    return _load_set('ron')


def load_ysi() -> Tuple[List[str], List[List[float]]]:
    """ loads yield sooting index data

    Returns:
        Tuple[List[str], List[List[float]]]: (SMILES, YSI values)
    """

    return _load_set('ysi')
