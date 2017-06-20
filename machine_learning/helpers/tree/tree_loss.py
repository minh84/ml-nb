import numpy as np


class TreeLoss(object):
    def __init__(self):
        pass


class TreeRegressionLoss(TreeLoss):
    '''
    In the case of regression, the tree loss function is the variance at each node i.e
        \sum_i (y_i - y_bar) ^2 = \sum_y y_i^2 - n_sample y_bar^2
    '''

    def __init__(self):
        self._sum = 0.  # store sum of y_i
        self._sum_Sqr = 0.  # store sum of y_i^2
        self._n_samples = 0  # store number of samples

        # shallow copy from outside shouldn't be changed
        self._samples_indices = None
        self._targets = None

        # node information
        self._start = None
        self._end = None

        # info for left & right node
        self._sum_left = 0.
        self._n_left = 0

        self._sum_right = 0.
        self._n_right = 0

        self._pos = None  # where to cut

    def init(self, targets, samples_indices, start, end):
        '''
        We init some computation before finding best-split
        Note that we keep samples_indices map such that a node will contain samples with
                samples_indices[start],...,samples_indices[end-1]
        The loss is variance, so we need to compute
            self._sum = sum y_i
            self._sum_Sqr = sum y_i^2
        :param targets: target values y_i
        :param samples_indices: a permutation of 0,...,N-1 which stores indices mapping of samples
        :param start: start index of a node
        :param end: end index of a nodes
        :return: None
        '''
        self._sum = 0.
        self._sum_Sqr = 0.
        self._n_samples = end - start

        # share a shallow copy
        self._targets = targets
        self._samples_indices = samples_indices
        self._n_totals = targets.shape[0]

        self._start = start
        self._end = end

        for i in range(start, end):
            idx = samples_indices[i]
            self._sum += targets[idx]
            self._sum_Sqr += targets[idx] ** 2

        self.reset()

    def reset(self):
        '''
        reset pos = start
        :return: 
        '''
        self._sum_left = 0.
        self._n_left = 0

        self._sum_right = self._sum
        self._n_right = self._n_samples
        self._pos = self._start

    def reverse_reset(self):
        '''
        reset pos = end
        :return: 
        '''
        self._sum_left = self._sum
        self._n_left = self._n_samples

        self._sum_right = 0.
        self._n_right = 0
        self._pos = self._end

    def update(self, new_pos):
        '''
        left_node : [self._start, new_pos)
        right_node: [new_pos, self._end)

        Note that we only update 
            self._sum_left & self._sum_right
        :param new_pos: 
        :return: 
        '''
        if (new_pos - self._pos <= self._end - self._pos):
            for i in range(self._pos, new_pos):
                idx = self._samples_indices[i]
                self._sum_left += self._targets[idx]
        else:
            self.reverse_reset()
            for i in range(self._end - 1, new_pos - 1, -1):
                idx = self._samples_indices[i]
                self._sum_left -= self._targets[idx]

        # compute left & right
        self._sum_right = self._sum - self._sum_left
        self._pos = new_pos
        self._n_left = self._pos - self._start
        self._n_right = self._end - self._pos

    def proxy_loss(self):
        '''
        Compute proxy_loss = cst - real_loss to save cpu time
        So in-order to minimize loss, we need to maximize proxy_loss
        :return: 
        '''
        retval = 0.
        if (self._n_left > 0):
            retval += self._sum_left ** 2.0 / self._n_left
        if (self._n_right > 0):
            retval += self._sum_right ** 2.0 / self._n_right

        return retval

    def node_loss(self):
        '''
        we compute loss of current node
        :return: 
        '''
        return self._sum_Sqr / self._n_samples - (self._sum / self._n_samples) ** 2.0

    def child_loss(self):
        '''
        we compute loss in child left & right node, the loss is mean-square-error
        :return: left_loss, right_loss
        '''
        left_loss = 0.
        right_loss = 0.
        left_Sqr = 0.
        for i in range(self._start, self._pos):
            idx = self._samples_indices[i]
            self._sum_left += self._targets[idx]
            left_Sqr += self._targets[idx]**2.0
        right_Sqr = self._sum_Sqr - left_Sqr

        if (self._n_left > 0):
            left_loss = left_Sqr / self._n_left - (self._sum_left / self._n_left) ** 2.0
        if (self._n_right > 0):
            right_loss = right_Sqr / self._n_right - (self._sum_right / self._n_right) ** 2.0

        return left_loss, right_loss

    def all_loss(self, best_pos):
        self.reset()
        self.update(best_pos)
        node_loss = self.node_loss()
        left_loss, right_loss = self.child_loss()

        loss_improvement = node_loss
        if (self._n_left > 0): loss_improvement -= left_loss * self._n_left / self._n_samples
        if (self._n_right > 0): loss_improvement -= right_loss * self._n_right / self._n_samples
        loss_improvement *= self._n_samples / self._n_totals

        return loss_improvement, node_loss, left_loss, right_loss