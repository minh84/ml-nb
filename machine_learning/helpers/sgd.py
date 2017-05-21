import numpy as np
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

