import os
from datetime import datetime
from time import time
import pickle
from typing import Callable, Optional, NoReturn, Sequence, Tuple, Any, Dict


# make logs, save objects 
class Logger:

    def __init__(self, exp_dir: str, log_fn: str, obj_ext: str='.pickle'):

        self.exp_dir = exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        self.log_fn = log_fn
        self.obj_ext = obj_ext

    # log details in the main logging file 
    def log(self, text: str) -> NoReturn:
        
        date = datetime.today()
        with open(self.log_fn, 'a') as f:
            f.write(f'{date.strftime("%Y-%m-%d %H:%M:%S")} {text}\n')

    # make logs in a diff file (used for saving intermediate variables during training)
    def write(self, text: str, fn: str) -> NoReturn:

        fn = os.path.join(self.exp_dir, fn)
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with open(fn, 'a') as f:
            f.write(text)

    # formatting tool
    def indent(self, s: str, indent: int = 0):

        return '  '*indent + s.replace('\n', '\n'+'  '*indent)

    # save objects as pickle files, unless a different extension is provided
    def save(self, obj: Any, fn: str) -> NoReturn:

        fn = os.path.join(self.exp_dir, fn)
        if not os.path.splitext(fn)[1]:
            fn += self.obj_ext
        os.makedirs(os.path.dirname(fn), exist_ok=True)

        with open(fn, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# time function calls and log them
class Timer:

    def __init__(self, logger: Logger, total_time: int = 0):

        self.total_time = total_time
        self.logger = logger

    # time the method with args and kwargs 
    def timeit(self, method: Callable, args: Optional[Sequence] = [], kwargs: Optional[Dict] = dict(), log_str: str = '') -> Tuple[Any, float]: 
        
        self.logger.log(f'Started {log_str}.')
        t = time()
        output = method(*args, **kwargs)
        run_time = time() - t
        self.logger.log(f'Finished {log_str}. Total time = {round(run_time, 6):.6f} seconds.')
        self.total_time += run_time 

        return output

    # reset timer
    def reset(self) -> NoReturn:

        self.total_time = 0