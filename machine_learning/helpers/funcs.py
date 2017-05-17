import numpy as np

def sigmoid(x):
    # clip x between -30 and 30
    x = np.maximum(-30, np.minimum(30, x))
    return 1.0/(1.0 + np.exp(-x))

def softmax(x):
    '''
    Compute softmax for x
    :param x: an 2d-array of shape N x D
    :return: softmax of x is 2d-array of shape N x D
        out[i,j] = exp(x[i,j]) / sum_j exp(x[i,j])
    '''
    assert (len(x.shape) == 2)

    x = np.exp(x - np.max(x, axis=1, keepdims=True))
    out = x / np.sum(x, axis=1, keepdims=True)
    return out