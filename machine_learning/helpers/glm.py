import numpy as np
from .funcs import sigmoid
from .sgd import Sgd, SgdMomentum
from .solver import solve

class Glm(object):
    def __init__(self):
        self._thetas = None

    def init_thetas(self, thetas):
        self._thetas = thetas

    def step(self, dtheta):
        self._thetas += dtheta

    def fit(self, train_X, train_y, val_X, val_y, epochs, batch_size,
            learning_rate = 1e-3, solver = 'SgdMomentum', use_intercept = True,
            debug=False):

        # add intercept
        if use_intercept:
            train_X = np.hstack([np.ones([train_X.shape[0], 1]), train_X])
            val_X = np.hstack([np.ones([val_X.shape[0], 1]), val_X])

        # initialized theta0 by a random-normal
        input_dim = train_X.shape[1]
        thetas_0 = np.random.randn(input_dim)

        if solver == 'Sgd':
            optimizer = Sgd(learning_rate=learning_rate)
        elif solver == 'SgdMomentum':
            optimizer = SgdMomentum(learning_rate=learning_rate)
        else:
            raise Exception('Unknown solver {}'.format(solver))

        self._dbg_loss = solve(self, optimizer, thetas_0, train_X, train_y, val_X, val_y,
                               epochs = epochs, batch_size = batch_size, debug = debug)

    def __call__(self, batch_X, batch_y):
        raise Exception('must be implemented in derived class')

    def error(self, batch_X, batch_y):
        raise Exception('must be implemented in derived class')

    def hypothesis(self, x):
        raise Exception('must be implemented in derived class')

class Lsm(Glm):
    def __init__(self):
        super(Lsm, self).__init__()

    def error(self, batch_X, batch_y):
        l2err, _ = self(batch_X, batch_y)
        return l2err

    def __call__(self, batch_X, batch_y):
        batch_size = batch_y.shape[0]
        err = self.hypothesis(batch_X) - batch_y
        loss = 0.5 * np.mean(np.square(err))
        dtheta = np.dot(batch_X.T, err) / batch_size

        return loss, dtheta

    def hypothesis(self, x):
        assert self._thetas is not None, 'model has NOT been fitted, self._thetas is still None'

        return x.dot(self._thetas)

class LogitReg(Glm):
    def __init__(self, reg = 1e-3):
        super(LogitReg, self).__init__()
        self._reg = reg

    def error(self, batch_X, batch_y):
        h_theta = self.hypothesis(batch_X)
        pred = np.zeros_like(batch_y)
        idx = np.where(h_theta > 0.5)[0]
        pred[idx] = 1

        return np.mean(pred != batch_y)

    def __call__(self, batch_X, batch_y):
        batch_size = batch_y.shape[0]

        h_theta = self.hypothesis(batch_X)
        loss = - np.mean(np.log(h_theta) * batch_y + np.log(1.0 - h_theta) * batch_y)
        dtheta = np.dot(batch_X.T, (h_theta - batch_y)) / batch_size

        # add regularisation
        loss += 0.5 * self._reg * np.sum(np.shape(self._thetas))
        dtheta += self._reg * self._thetas

        return loss, dtheta

    def hypothesis(self, x):
        assert self._thetas is not None, 'model has NOT been fitted, self._thetas is still None'

        z = x.dot(self._thetas)
        return sigmoid(z)

def demo_bin_class(num_samples = 100):
    '''
    generate demo dataset for binary classification
    :param num_samples: number of samples
    :return: 
        X: features X and labels y
    '''
    eps = 0.1
    X = []
    y = []
    while len(X) < num_samples:
        x = np.random.uniform(low = 0.0, high=1.0, size=[2])
        if x[0] + x[1] - 1.0 > eps:
            X.append(x)
            y.append(1)
        elif x[0] + x[1] - 1.0 < -eps:
            X.append(x)
            y.append(0)

    return np.array(X), np.array(y)