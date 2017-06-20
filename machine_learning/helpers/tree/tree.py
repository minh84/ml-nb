import numpy as np
import collections
from .tree_loss import TreeRegressionLoss
from .tree_splitter import Splitter, SplitRecord

class Node(object):
    def __init__(self, feat_idx, split_level):
        self._feat_idx    = feat_idx
        self._split_level = split_level

        self._left_child  = None
        self._right_child = None

    def __str__(self):
        return '({:d}, {:f}) left[{}] right[{}]'.format(self._feat_idx, self._split_level,
                                                            self._left_child, self._right_child)

def compute_split_node(splitter, start, end, depth):
    record = splitter.node_split(start, end)
    return (record, depth)

class PriorityHeap(object):
    def __init__(self, cmp_func):
        self._nodes = []
        self._cmp_func = cmp_func

    def is_empty(self):
        return len(self._nodes)==0

    def heapify_up(self, pos):
        '''
        maintain max-heap property by moving up node[pos]
        this is used when we add a new node or change value of one node
            while node[parent[pos]] < node[pos] do
                swap node[paraent[pos]] with node[pos]
        :param pos: 
        :return: 
        '''
        if pos > 0:
            parent_pos = self.parent(pos)
            while (pos > 0 and self._cmp_func(self._nodes[parent_pos], self._nodes[pos]) < 0):
                # swap
                self._nodes[parent_pos], self._nodes[pos] = self._nodes[pos], self._nodes[parent_pos]
                pos = parent_pos
                parent_pos = self.parent(pos)

    def parent(self, pos):
        return (pos-1)//2

    def left(self, pos):
        return 2*pos + 1

    def right(self, pos):
        return 2*(pos + 1)

    def heapify_down(self, pos):
        '''
        maintain max-heap property by moving down node[pos]
        assume tree rooted at left(pos) and right(pos) are already max-heap
        but node[pos] might be smaller than node[left(pos)] or node[right(pos)]
        we allow to move node[pos] down
        :param pos: 
        :return: 
        '''
        largest = pos
        l = self.left(pos)
        r = self.right(pos)
        if (l < len(self._nodes) and self._cmp_func(self._nodes[largest], self._nodes[l]) < 0):
            largest = l
        if (r < len(self._nodes) and self._cmp_func(self._nodes[largest], self._nodes[r]) < 0):
            largest = r

        if (largest != pos):
            # exchange pos & largest
            self._nodes[pos], self._nodes[largest] = self._nodes[largest], self._nodes[pos]
            # continue on down stream tree
            self.heapify_down(largest)

    def push(self, node):
        self._nodes.append(node)
        self.heapify_up(len(self._nodes) - 1)

    def pop_max(self):
        '''        
        :return: 
        '''
        if self.is_empty():
            raise Exception('Can NOT pop empty heap')

        retval = self._nodes[0]

        # swap last element to position 0 then heapify_down
        if len(self._nodes) > 1:
            self._nodes[0] = self._nodes.pop()
            self.heapify_down(0)
        else:
            self._nodes = []

        return retval

    def peak_max(self):
        return self._nodes[0]

def cmp_split_node(split_node_l, split_node_r):
    _, split_record_l, _ = split_node_l
    _, split_record_r, _ = split_node_r

    return split_record_l.improvement - split_record_r.improvement

class RegressorTree(object):
    def __init__(self,
                 max_depth,
                 min_samples_split = 2,
                 min_samples_leaf = 1):
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

        # tree nodes is stored in an array
        # self._nodes[parent]._left_child => id of left child of self._nodes[parent]
        self._nodes = []

    def fit(self, X, y):
        # tree algorithm is done as below
        #   0) create a root node
        #   1) compute best split, put (root, best_split) in queue

        loss_func = TreeRegressionLoss()
        splitter  = Splitter(loss_func,
                             min_samples_split=self._min_samples_split,
                             min_samples_leaf=self._min_samples_leaf)

        splitter.init(X, y)
        ph = PriorityHeap(cmp_split_node)
        n_samples = X.shape[0]

        # get root node
        node_id, split_node, depth = self.add_split_node(splitter, 0, n_samples, 0, None, None)

        if node_id != -1:
            ph.push((node_id, split_node, depth))

        while not ph.is_empty():
            node_id, split_node, depth = ph.pop_max()

            # get left split
            left_id, split_left, left_depth    = self.add_split_node(splitter, split_node.start, split_node.pos, depth+1, node_id, True)
            right_id, split_right, right_depth = self.add_split_node(splitter, split_node.pos, split_node.end, depth+1, node_id, False)

            if left_id != -1:
                ph.push((left_id, split_left, left_depth))
            if right_id != -1:
                ph.push((right_id, split_right, right_depth))

    def add_split_node(self, splitter, start, end, depth, parent_id, is_left):
        if depth >= self._max_depth:
            return (-1, None, None)
        # get split node
        split_node = splitter.node_split(start, end)
        node_id = -1

        if split_node is not None:
            new_node = Node(split_node.feat_idx, split_node.split_level)

            node_id = len(self._nodes)
            self._nodes.append(new_node)

            if parent_id is not None:
                if is_left:
                    self._nodes[parent_id].left_child = node_id
                else:
                    self._nodes[parent_id].right_child = node_id
        return (node_id, split_node, depth)




