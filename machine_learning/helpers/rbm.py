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

        # use xavier init
        bound = 4.0 * np.sqrt(6. / (self._input_dim + self._hidden_dim))
        self._vh_weights = np.random.uniform(low = -bound, high = bound,
                                             size=[self._input_dim, self._hidden_dim])

    def visible_to_hidden_probs(self, visible_state):
        hidden_probs = sigmoid(visible_state.dot(self._vh_weights) + self._h_biases)
        return hidden_probs

    def hidden_to_visible_probs(self, hidden_state):
        visible_probs = sigmoid(hidden_state.dot(self._vh_weights.T) + self._v_biases)
        return visible_probs

    def gibbs_v2h(self, v):
        h_probs = self.visible_to_hidden_probs(v)
        h = sample_bin_from_prob(h_probs)
        return h_probs, h

    def gibbs_h2v(self, h):
        v_probs = self.hidden_to_visible_probs(h)
        v = sample_bin_from_prob(v_probs)
        return v_probs, v

    def gibbs_hvh(selfs, h):
        v_probs, v = selfs.gibbs_h2v(h)
        h_probs, h = selfs.gibbs_v2h(v)
        return v_probs, v, h_probs, h

    def gibbs_vhv(selfs, v):
        h_probs, h = selfs.gibbs_v2h(v)
        v_probs, v = selfs.gibbs_h2v(h)

        return h_probs, h, v_probs, v

    def cdk(self, visible_state, cd_steps = 1, persistent = None):
        # since visibale_state can be image intensity which has value in [0,1]
        # we sample a binary value from it
        v0 = sample_bin_from_prob(visible_state)

        h0_probs, h0 = self.gibbs_v2h(v0)

        # fantasy particle either start from h0 or persistent
        if persistent is None:
            h = h0
        else:
            h = persistent

        v = None
        for i in range(cd_steps):
            v_probs, v, h_probs, h = self.gibbs_hvh(h)


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
                if use_pcd:
                    persisten = self.step(batch_data, learning_rate, cd_steps=cd_steps, persistent=persisten)
                else:
                    _ = self.step(batch_data, learning_rate, cd_steps=cd_steps)

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
        v0 = sample_bin_from_prob(v0_prob)

        for i in range(num_gibbs_step):
            h_probs, h, v_probs, v = self.gibbs_vhv(v0)
        return v_probs, v