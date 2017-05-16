import numpy as np
import sys

from .funcs import sigmoid

def sample_bin_from_prob(probs):
    output = (np.random.uniform(size=probs.shape) < probs).astype(probs.dtype)
    return output

def goodness_grad(visible_state, hidden_state):
    # get number of samples
    N = visible_state.shape[0]

    dW_vh = np.dot(visible_state.T, hidden_state) / N
    db_v  = np.mean(visible_state, axis=0)
    db_h  = np.mean(hidden_state, axis=0)
    return dW_vh, db_v, db_h

class RBMachine(object):
    def __init__(self, input_dim, hidden_dim):
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim

        # build model
        self.build_model()

    def build_model(self):
        self._v_biases = np.zeros([self._input_dim])
        self._h_biases = np.zeros([self._hidden_dim])
        self._vh_weights = np.random.randn(self._input_dim, self._hidden_dim)

    def visible_to_hidden_probs(self, visible_state):
        hidden_probs = sigmoid(visible_state.dot(self._vh_weights) + self._h_biases)
        return hidden_probs

    def hidden_to_visible_probs(self, hidden_state):
        visible_probs = sigmoid(hidden_state.dot(self._vh_weights.T) + self._v_biases)
        return visible_probs

    def gibbs_v2h(self, v_probs):
        v = sample_bin_from_prob(v_probs)
        h_probs = self.visible_to_hidden_probs(v)
        return v, h_probs

    def gibbs_h2v(self, h_probs):
        h = sample_bin_from_prob(h_probs)
        v_probs = self.hidden_to_visible_probs(h)
        return h, v_probs

    def gibbs_hvh(selfs, h_probs):
        h, v_probs = selfs.gibbs_h2v(h_probs)
        v, h_probs = selfs.gibbs_v2h(v_probs)
        return h, v, h_probs

    def cdk(self, visible_state, cd_steps = 1, persistent = None):
        # since visibale_state can be image intensity which has value in [0,1]
        # we sample a binary value from it
        v0, h0_probs = self.gibbs_v2h(visible_state)
        h0 = None
        v  = None
        if persistent is None:
            h_probs = h0_probs
        else:
            h_probs = persistent

        for i in range(cd_steps):
            h, v, h_probs = self.gibbs_hvh(h_probs)
            if (i==0):
                h0 = h
        h = sample_bin_from_prob(h_probs)

        dW_vh_0, db_v_0, db_h_0 = goodness_grad(v0, h0)
        dW_vh_k, db_v_k, db_h_k = goodness_grad(v, h)

        # we compute CD1 update
        dW_vh = dW_vh_0 - dW_vh_k
        db_v  = db_v_0 - db_v_k
        db_h  = db_h_0 - db_h_k

        persistent = h

        return dW_vh, db_v, db_h, persistent

    def step(self, batch_data, learning_rate, cd_steps = 1, persistent = None):
        dW_vh, db_v, db_h, persistent = self.cdk(batch_data, cd_steps = cd_steps, persistent = persistent)

        self._v_biases += learning_rate * db_v
        self._h_biases += learning_rate * db_h
        self._vh_weights += learning_rate * dW_vh

        return persistent

    def get_batches(self, train_data, batch_size):
        N = train_data.shape[0]
        idx = np.arange(N)
        nb_batch = N // batch_size
        np.random.shuffle(idx)
        for i in range(nb_batch):
            yield train_data[idx[i*batch_size: (i+1)*batch_size]]

    def train(self, train_data, validation_data, epochs, batch_size, learning_rate = 1e-3, prive_every=50, cd_steps = 1, use_pcd = False):
        steps = 0
        persisten = None
        for i in range(epochs):
            for batch_data in self.get_batches(train_data, batch_size):
                persisten = self.step(batch_data, learning_rate, cd_steps = cd_steps, persistent = persisten)
                if not use_pcd:
                    persisten = None
                steps += 1
                if steps % prive_every == 0:
                    loss = self.loss(validation_data)
                    sys.stdout.write("\rEpoch ({}/{})".format(i + 1, epochs)
                                     + "Step {:>5d} Loss {:.4f} ".format(steps, loss))
            print("\n")

    def loss(self, validation_data):
        v0, h0_probs = self.gibbs_v2h(validation_data)
        h0, v1_probs = self.gibbs_h2v(h0_probs)

        return np.mean(np.square(validation_data - v1_probs))

    def sample(self, v0_prob, num_gibbs_step = 500):
        v_prob = v0_prob
        h_prob = None

        v = None
        h = None
        for i in range(num_gibbs_step):
            v, h_prob = self.gibbs_v2h(v_prob)
            h, v_prob = self.gibbs_h2v(h_prob)
        return v, v_prob