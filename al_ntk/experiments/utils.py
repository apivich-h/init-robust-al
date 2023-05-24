import os
from datetime import datetime
from typing import Dict

from omegaconf import DictConfig, OmegaConf


def init_dir(dir: str, append: str = '') -> str:

    date = datetime.today()
    par_dir = os.path.join(dir, date.strftime('%d-%m-%Y-%H-%M-%S')) + append
    os.makedirs(par_dir, exist_ok=True)

    return par_dir

def get_kwargs(d: DictConfig, update: Dict = dict()) -> Dict:

    d = OmegaConf.to_container(d) if d else dict()
    d.update(update)

    return d