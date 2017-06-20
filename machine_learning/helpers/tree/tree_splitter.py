import numpy as np
import collections

'''
We create a type to hold split-record
    * feat_idx: which feature is used in the split
    * pos: the split's position
    * split_level: threshold for the split
    * improvement: improve of impurity after the split
    * impurity_left: impurity of the left node
    * impurity_right: impurity of the right node
'''
SplitRecord = collections.namedtuple('SplitRecord',
                                     ['feat_idx',
                                      'start',
                                      'end',
                                      'pos',
                                      'split_level',
                                      'improvement',
                                      'node_loss',
                                      'left_loss',
                                      'right_loss'])

FEATURE_THRESHOLD = 1e-7

class Splitter(object):
    def __init__(self,
                 loss_func,
                 min_samples_split = 2,
                 min_samples_leaf = 1):

        # loss function
        self._loss_func = loss_func

        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def init(self, X, y):
        '''
        Initialization the Splitter object to fit features X & targets y
        :param X: features is 2d-array (n_samples, n_features)
        :param y: target is 1d-array (n_samples)
        :return: 
        '''
        self._X = X
        self._y = y

        self._n_samples, self._n_features = X.shape
        self._samples_indices = np.arange(self._n_samples)

        # presort index on features
        self._X_sorted_idx = np.argsort(self._X, axis=0)

    def node_split(self, start, end):
        '''
        Compute a node-split for a node that hold 
            self._samples_indices[start:end] samples
        :param start: start index of the current node 
        :param end: end index of the current node
        :return: 
            SplitRecord: to record how to split information
            or None if we can't split further
        '''

        # init loss-function to pre-compute some variables
        self._loss_func.init(self._y, self._samples_indices, start, end)

        # compute split node
        sample_mask = set([self._samples_indices[p] for p in range(start, end)])
        n_samples = end - start

        # this must be a leaf node
        if (n_samples < self._min_samples_split or n_samples < 2*self._min_samples_leaf):
            return None

        # store impurity improvement
        node_impurity = self._loss_func.proxy_loss()
        best_impurity = node_impurity
        best_feat = None
        best_pos = None

        # loop through each feature to compute the split location
        n_constant = 0
        min_p = start + self._min_samples_leaf
        for feat_idx in range(self._n_features):
            # first get sorted index for current feature
            sorted_idx = [p for p in self._X_sorted_idx[:, feat_idx] if p in sample_mask]
            Xf = self._X[sorted_idx, feat_idx]

            # we consider this as constant feature
            if Xf[-1] < Xf[0] + FEATURE_THRESHOLD:
                n_constant += 1
                continue

            # update sample indices to be sorted
            self._samples_indices[start:end] = sorted_idx
            # reset loss func (since it might be modified in previous feat_idx)
            self._loss_func.reset()


            for p in range(start + 1, end - self._min_samples_leaf + 1):
                if Xf[p-start] < Xf[p-1-start] + FEATURE_THRESHOLD:
                    continue

                if (p >= min_p):
                    # compute loss: left node [0, p) right node [p, n_sample)
                    self._loss_func.update(p)
                    current_impurity = self._loss_func.proxy_loss()

                    if current_impurity > best_impurity:
                        best_impurity = current_impurity
                        best_feat = feat_idx
                        best_pos = p

        if best_feat is not None:
            sorted_idx = [p for p in self._X_sorted_idx[:, best_feat] if p in sample_mask]

            # update sample indices to be sorted in best_feat direction
            self._samples_indices[start:end] = sorted_idx

            # compute loss for left-node & right-node
            loss_improvement, node_loss, left_loss, right_loss = self._loss_func.all_loss(best_pos)

            Xf = self._X[sorted_idx, best_feat]
            split_level = (Xf[best_pos-1-start] + Xf[best_pos-start])/2.0
            return SplitRecord(best_feat,
                               start,
                               end,
                               best_pos,
                               split_level,
                               improvement=loss_improvement,
                               node_loss=node_loss,
                               left_loss=left_loss,
                               right_loss=right_loss)
        else:
            return None