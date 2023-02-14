import numpy as np


def dict_argmax(d):
    """
    Return the key with the max value from a given dict

    """
    v = np.array(list(d.values()))
    k = np.array(list(d.keys()))

    return k[np.random.choice(np.where(v == v.max())[0])]
