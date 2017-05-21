import numpy as np
from .funcs import sigmoid, get_batch_indices

class Glm(object):
    def __init__(self):
        self._thetas = None

    def fit(self, train_X, train_y, val_X, val_y,
            batch_size, epochs, learning_rate = 1e-3, debug=False):
        # get number of sample and input dimension
        N, input_dim = train_X.shape
        nb_batches = N // batch_size

        # add intercept
        train_X = np.hstack([np.ones([N, 1]), train_X])
        val_X = np.hstack([np.ones([val_X.shape[0], 1]), val_X])
        input_dim += 1

        # initialized by a random-normal
        self._thetas = np.random.randn(input_dim)
        self._momentum = np.zeros_like(self._thetas)
        self._dbg_loss = None
        # if debug on, we store loss
        if debug:
            self._dbg_loss = []

        # iteratively update theta
        for e in range(epochs):
            for idx in get_batch_indices(N, batch_size):
                # get batch data
                batch_X = train_X[idx]
                batch_y = train_y[idx]

                # sgd update
                self._step(batch_X, batch_y, learning_rate)
                if debug:
                    train_err = self.error(batch_X, batch_y)
                    val_err = self.error(val_X, val_y)
                    self._dbg_loss.append([train_err, val_err])

    def _step(self, batch_X, batch_y, learning_rate):
        _, dtheta = self(batch_X, batch_y)

        # use sgd with momentum
        self._momentum = 0.9 * self._momentum + dtheta
        self._thetas = self._thetas - learning_rate * self._momentum

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
    def __init__(self):
        super(LogitReg, self).__init__()

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