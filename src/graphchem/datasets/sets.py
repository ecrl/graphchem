from csv import DictReader
from os import path

import torch

_CSV_PATH = path.join(path.dirname(path.abspath(__file__)), "static")


def _load_set(prop: str) -> tuple[list[str], torch.Tensor]:
    """
    Loads data for a given property from a CSV file.

    Parameters
    ----------
    prop : str
        The property to obtain data for.

    Returns
    -------
    Tuple[List[str], torch.Tensor]
        A tuple containing two elements:
        - List[str]: A list of SMILES strings.
        - torch.Tensor: A tensor of property values with dtype float32.
    """
    filename = path.join(_CSV_PATH, f"{prop}.csv")
    with open(filename) as csv_file:
        reader = DictReader(csv_file)
        rows = [r for r in reader]
    csv_file.close()
    return (
        [r["SMILES"] for r in rows],
        torch.tensor([[float(r[f"{prop.upper()}"])] for r in rows]).type(torch.float32),
    )


def load_cn() -> tuple[list[str], torch.Tensor]:
    """
    Loads cetane number data.

    Returns
    -------
    Tuple[List[str], torch.Tensor]
        A tuple containing two elements:
        - List[str]: A list of SMILES strings.
        - torch.Tensor: A tensor of CN values with dtype float32.
    """
    return _load_set("cn")


def load_lhv() -> tuple[list[str], torch.Tensor]:
    """
    Loads lower heating value data.

    Returns
    -------
    Tuple[List[str], torch.Tensor]
        A tuple containing two elements:
        - List[str]: A list of SMILES strings.
        - torch.Tensor: A tensor of LHV values with dtype float32.
    """
    return _load_set("lhv")


def load_mon() -> tuple[list[str], torch.Tensor]:
    """
    Loads motor octane number data.

    Returns
    -------
    Tuple[List[str], torch.Tensor]
        A tuple containing two elements:
        - List[str]: A list of SMILES strings.
        - torch.Tensor: A tensor of MON values with dtype float32.
    """
    return _load_set("mon")


def load_ron() -> tuple[list[str], torch.Tensor]:
    """
    Loads research octane number data.

    Returns
    -------
    Tuple[List[str], torch.Tensor]
        A tuple containing two elements:
        - List[str]: A list of SMILES strings.
        - torch.Tensor: A tensor of RON values with dtype float32.
    """
    return _load_set("ron")


def load_ysi() -> tuple[list[str], torch.Tensor]:
    """
    Loads yield sooting index data.

    Returns
    -------
    Tuple[List[str], torch.Tensor]
        A tuple containing two elements:
        - List[str]: A list of SMILES strings.
        - torch.Tensor: A tensor of YSI values with dtype float32.
    """
    return _load_set("ysi")
