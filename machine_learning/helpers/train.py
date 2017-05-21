import numpy as np
from .funcs import get_batch_indices

class Sgd(object):
    def __init__(self, learning_rate = 1e-3):
        self._learning_rate = learning_rate

    def minimize(self, cost_fn):
        '''
        this function set cost function to be minimized
        :param cost_fn: a functor that has a member function loss(batch_X, batch_y)
        :return: 
        '''
        self._cost_fn = cost_fn


    def step(self, batch_X, batch_y):
        _, dtheta = self._cost_fn(batch_X, batch_y)

        self._cost_fn.step(-self._learning_rate*dtheta)

class SgdMomentum(Sgd):
    def __init__(self, learning_rate = 1e-3, moment = 0.9):
        self._momentum = None
        self._moment = moment

        super(SgdMomentum, self).__init__(learning_rate)

    def step(self, batch_X, batch_y):
        _, dtheta = self._cost_fn(batch_X, batch_y)
        if self._momentum is None:
            self._momentum = np.zeros_like(dtheta)

        self._momentum = self._moment * self._momentum + dtheta

        self._cost_fn.step(-self._learning_rate * self._momentum)

def solve(model, optimizer, thetas_0,
          train_X, train_y, val_X, val_y,
          epochs, batch_size, debug = False):

    optimizer.minimize(model)
    model.init_thetas(thetas_0)
    dbg_loss = None

    if debug:
        dbg_loss = []
    N = train_X.shape[0]
    for e in range(epochs):
        for idx in get_batch_indices(N, batch_size):
            batch_X = train_X[idx]
            batch_y = train_y[idx]

            # run optimizer update
            optimizer.step(batch_X, batch_y)

            # store error if debug = True
            if debug:
                train_err = model.error(batch_X, batch_y)
                valid_err = model.error(val_X, val_y)
                dbg_loss.append([train_err, valid_err])

    print (len(dbg_loss))
    return dbg_loss