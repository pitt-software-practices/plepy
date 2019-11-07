import numpy as np


def sigfig(n, sf: int):
    mag = int(np.floor(np.log10(n)))
    rf = sf - mag - 1
    return round(n, rf)
