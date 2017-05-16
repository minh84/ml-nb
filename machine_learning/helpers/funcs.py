import numpy as np

def sigmoid(x):
    # clip x between -30 and 30
    x = np.maximum(-30, np.minimum(30, x))
    return 1.0/(1.0 + np.exp(-x))