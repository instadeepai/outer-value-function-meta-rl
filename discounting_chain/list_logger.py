import time
from typing import Dict, List, Union

import numpy as np


class ListLogger:
    def __init__(self):
        self.history: Dict[str, List[Union[np.ndarray, float, int]]] = {}
        self.iter_n = 0
        self.start_time = 0.0

    def write(self, data: Dict) -> None:
        for key, value in data.items():
            if key in self.history:
                self.history[key].append(value)
            else:
                self.history[key] = [value]

    def close(self) -> None:
        pass

    def init_time(self) -> None:
        self.start_time = time.time()
