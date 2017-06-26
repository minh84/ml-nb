import numpy as np
import collections
from .tree_loss import TreeRegressionLoss
from .tree_splitter import Splitter

class Node(object):
    def __init__(self, feat_idx, split_level, n_samples, value, mse):
        self._feat_idx    = feat_idx
        self._split_level = split_level
        self._n_samples = n_samples
        self._value = value
        self._mse = mse

        self._left_child  = None
        self._right_child = None

    def __str__(self):
        retval= '(sample={:d}, value={:.4f}, mse={:.4f}) split=({},{})'.format(self._n_samples, self._value,
                                                                           self._mse, self._feat_idx, self._split_level)
        if self._left_child is not None:
            retval += ' left-child[{}]'.format(self._left_child)

        if self._right_child is not None:
            retval += ' right-child[{}]'.format(self._right_child)

        return retval

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

            # get left & right split
            left_id, split_left, left_depth    = self.add_split_node(splitter,
                                                                     split_node.start,
                                                                     split_node.pos,
                                                                     depth+1,
                                                                     node_id,
                                                                     True)

            right_id, split_right, right_depth = self.add_split_node(splitter,
                                                                     split_node.pos,
                                                                     split_node.end,
                                                                     depth+1,
                                                                     node_id,
                                                                     False)

            if left_id != -1:
                ph.push((left_id, split_left, left_depth))
            if right_id != -1:
                ph.push((right_id, split_right, right_depth))

    def add_split_node(self, splitter, start, end, depth, parent_id, is_left):
        split_node = None
        feat_idx = None
        split_level = None
        split_node_id = -1
        n_samples, value, mse = splitter.node_value(start, end)

        if depth < self._max_depth:
            # get split node
            split_node = splitter.node_split(start, end)

            if split_node is not None:
                feat_idx, split_level = split_node.feat_idx, split_node.split_level
                split_node_id = len(self._nodes)

        # can't compute split-node
        node_id = len(self._nodes)
        self._nodes.append(Node(feat_idx, split_level, n_samples, value, mse))

        if parent_id is not None:
            if is_left:
                self._nodes[parent_id]._left_child = node_id
            else:
                self._nodes[parent_id]._right_child = node_id

        return (split_node_id, split_node, depth)

    def export_graphviz(self, feat_names):
        retval = '''digraph Tree {
node [shape=box, style="rounded", color="black", fontname=helvetica] ;
edge [fontname=helvetica] ;\n'''
        for i, n in enumerate(self._nodes):
            node_str = 'mse = {:.4f}<br/>samples = {}<br/>value = {:.4f}'.format(n._mse, n._n_samples, n._value)
            if n._feat_idx is not None:
                line = '{} [label=<{} &le; {:.4f}<br/>{}>] ;\n'.format(i, feat_names[n._feat_idx], n._split_level, node_str)
            else:
                line = '{} [label=<{}>]\n'.format(i, node_str)

            retval += line

            if n._left_child:
                if i == 0:
                    retval += '{} -> {} [labeldistance=2.5, labelangle=45, headlabel="True"] ;\n'.format(i, n._left_child)
                else:
                    retval += '{} -> {} ;\n'.format(i, n._left_child)
            if n._right_child:
                if i == 0:
                    retval += '{} -> {} [labeldistance=2.5, labelangle=-45, headlabel="False"] ;\n'.format(i, n._right_child)
                else:
                    retval += '{} -> {} ;\n'.format(i, n._right_child)

        retval += '}'

        return retval