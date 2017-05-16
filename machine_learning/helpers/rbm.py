import numpy as np
import sys

from .funcs import sigmoid

def sample_bin_from_prob(probs):
    output = (np.random.uniform(size=probs.shape) < probs).astype(probs.dtype)
    return output

def goodness_grad(visible_state, hidden_state):
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

    def cd1(self, visible_state):
        # since visibale_state can be image intensity which has value in [0,1]
        # we sample a binary value from it
        v0, h0_probs = self.gibbs_v2h(visible_state)
        h0, v1_probs = self.gibbs_h2v(h0_probs)
        v1, h1_probs = self.gibbs_v2h(v1_probs)
        h1 = sample_bin_from_prob(h1_probs)

        dW_vh_0, db_v_0, db_h_0 = goodness_grad(v0, h0)
        dW_vh_1, db_v_1, db_h_1 = goodness_grad(v1, h1)

        # we compute update
        dW_vh = dW_vh_0 - dW_vh_1
        db_v = db_v_0 - db_v_1
        db_h = db_h_0 - db_h_1

        return dW_vh, db_v, db_h

    def step(self, batch_data, learning_rate):
        dW_vh, db_v, db_h = self.cd1(batch_data)

        self._v_biases += learning_rate * db_v
        self._h_biases += learning_rate * db_h
        self._vh_weights += learning_rate * dW_vh

    def get_batches(self, train_data, batch_size):
        N = train_data.shape[0]
        idx = np.arange(N)
        nb_batch = N // batch_size
        np.random.shuffle(idx)
        for i in range(nb_batch):
            yield train_data[idx[i*batch_size: (i+1)*batch_size]]

    def train(self, train_data, validation_data, epochs, batch_size, learning_rate = 1e-3, prive_every=50):
        steps = 0
        for i in range(epochs):
            for batch_data in self.get_batches(train_data, batch_size):
                self.step(batch_data, learning_rate)
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

