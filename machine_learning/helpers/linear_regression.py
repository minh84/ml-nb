import numpy as np
from scipy import linalg
from tabulate import tabulate

# helper function to generate data
def create_dataset(theta, min_x, max_x, sigma, N):
    '''
    We generate N sample randomly uniform between min_x, max_x
    then we append 1 to each sample
                  y[i] = x[i] * theta' + eps[i]
    where eps[i] ~ N(0, sigma^2)

    Input arguments
    :param theta: a ndarray of shape [D+1]
    :param min_x: min value for x of shape [D]
    :param max_x: max value for x of shape [D]
    :param sigma: standard deviation for error
    :param N: number of sample

    :return:
        X: a matrix of shape [N, D+1] (we append 1 at the beginning)
        y: a ndarray of shape [N]
    '''
    D = len(theta) - 1
    sample_x = np.random.uniform(low=min_x, high=max_x, size=[N, D])
    X = np.hstack([np.ones([N, 1]), sample_x])
    y = X.dot(theta) + sigma * np.random.randn(N)
    return X, y

class LinearRegressionModel(object):
    def __init__(self):
        self._params = None
        self._residual = None
        self._stats = None

    def fit(self, X, y):
        XTX = np.dot(X.transpose(), X)
        XTy = np.dot(X.transpose(), y)

        self._params = linalg.solve(XTX, XTy, assume_a = 'sym')
        self._residual = y - np.dot(X, self._params)

        # compute some statistics
        v = np.maximum(1.0e-10, np.diag(linalg.inv(XTX)))   #ensure positive
        self._stats = {'sigma' : np.sqrt(np.var(self._residual)),
                       'v'     : v,
                       'R2'    : 1.0 - np.var(self._residual) / np.var(y)}

    def summary(self):
        assert (self._params is not None)
        print ('R2-score {:.3f}\n'.format(self._stats['R2']))

        summary_table =  []
        summary_headers = ['coef', 'fitted', 'F-score', 'low 95%', 'high 95%']
        for i in range(len(self._params)):
            stddev = self._stats['sigma'] * np.sqrt(self._stats['v'][i])
            fscore = self._params[i]/stddev
            low = self._params[i] - 1.96 * stddev
            high = self._params[i] + 1.96 * stddev
            row = ['theta_{}'.format(i), self._params[i], fscore, low, high]
            summary_table.append(row)
        print('Fitted parameters')
        print (tabulate(summary_table, headers = summary_headers))

def create_poly_dataset(theta, min_x, max_x, sigma, N, deg = None):
    x_val = np.random.uniform(low=min_x, high=max_x, size=[N])
    if deg is None:
        deg = len(theta) - 1
    X = np.polynomial.polynomial.polyvander(x_val, deg)
    y = np.polyval(theta, x_val) + sigma * np.random.randn(N)
    return X,y

class RidgeRegressionModel(object):
    def __init__(self, reg = 0.):
        assert (reg >= 0.0)
        self._reg = reg
        self._params = None

    def fit(self, X, y):
        XTX = np.dot(X.transpose(), X)
        XTy = np.dot(X.transpose(), y)

        D = XTX.shape[0]
        self._params = linalg.solve(XTX + self._reg * np.eye(D), XTy, assume_a='sym')

    def predict(self, x):
        return np.polyval(self._params, x)