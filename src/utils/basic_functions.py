import os
import yaml
from pathlib import Path
from typing import Dict, Tuple


# folder to load config file
root = Path(__file__).parents[2]
CONFIG_FILE = os.path.join(root, 'config.yaml')


# read config params for current project
def get_config(fld: str) -> Dict:
    # read yaml file
    with open(fld) as file:
        yaml_data = yaml.safe_load(file)

    return yaml_data


# string to int
def aninteger(s: str) -> int:
    return int(s)


# string to tuple
def atuple(s: str) -> Tuple[int, int]:
    a_tuple = tuple(int(num) for num in s.replace('(', '').replace(')', '').replace('...', '').split(', '))
    return a_tuple


# string to float
def afloat(s: str) -> float:
    return float(s)
