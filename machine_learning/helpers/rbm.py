import numpy as np
import sys

from .funcs import sigmoid, softmax

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

        # use momentum
        self._momentum_dv   = np.zeros_like(self._v_biases)
        self._momentum_dh   = np.zeros_like(self._h_biases)
        self._momentum_dWvh = np.zeros_like(self._vh_weights)

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

    def step(self, batch_data, learning_rate, cd_steps = 1, persistent = None, use_momentum = True):
        dW_vh, db_v, db_h, persistent = self.cdk(batch_data, cd_steps = cd_steps, persistent = persistent)

        if use_momentum:
            self._momentum_dv = 0.9 * self._momentum_dv + db_v
            self._momentum_dh = 0.9 * self._momentum_dh + db_h
            self._momentum_dWvh = 0.9 * self._momentum_dWvh + dW_vh
        else:
            self._momentum_dv   = db_v
            self._momentum_dh   = db_h
            self._momentum_dWvh = dW_vh

        self._v_biases += learning_rate * self._momentum_dv
        self._h_biases += learning_rate * self._momentum_dh
        self._vh_weights += learning_rate * self._momentum_dWvh

        return persistent

    def get_batch_indices(self, N, batch_size):
        idx = np.arange(N)
        nb_batch = N // batch_size
        np.random.shuffle(idx)
        for i in range(nb_batch):
            yield idx[i*batch_size: (i+1)*batch_size]

    def train(self, train_data, validation_data, epochs, batch_size, learning_rate = 1e-3, prive_every=50, cd_steps = 1, use_pcd = False):
        steps = 0
        persisten = None
        N = train_data.shape[0]
        for i in range(epochs):
            for batch_idx in self.get_batch_indices(N, batch_size):
                batch_data = train_data[batch_idx]
                if use_pcd:
                    persisten = self.step(batch_data, learning_rate, cd_steps=cd_steps, persistent=persisten, use_momentum = False)
                else:
                    _ = self.step(batch_data, learning_rate, cd_steps=cd_steps, use_momentum = True)

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

    def sample_from_h(self, h0_prob, num_gibbs_step = 500):
        h0 = sample_bin_from_prob(h0_prob)

        for i in range(num_gibbs_step):
            v_probs, v, h_probs, h = self.gibbs_hvh(h0)
        return v_probs, v

    def train_classification(self, train_data, train_label, valid_data, valid_label,
                             epochs, batch_size, learning_rate = 1e-3, print_every = 50):
        N, num_class = train_label.shape
        Wout = np.random.randn(self._hidden_dim, num_class) * np.sqrt(self._hidden_dim)
        bout = np.zeros([num_class])
        steps = 0
        train_rep = self.visible_to_hidden_probs(train_data)
        valid_rep = self.visible_to_hidden_probs(valid_data)
        nb_epochs = epochs//20

        momentum_W = np.zeros_like(Wout)
        momentum_b = np.zeros_like(bout)
        for i in range(epochs):
            for batch_idx in self.get_batch_indices(N, batch_size):
                h_rep = train_rep[batch_idx]
                label_data = train_label[batch_idx]

                # compute logits & class_probs
                logits = np.dot(h_rep, Wout) + bout
                class_probs = softmax(logits)

                # compute loss
                loss = - np.mean(np.sum(np.log(class_probs + 1e-10) * label_data, axis=1))
                # compute gradient
                dloss_dlogits = (class_probs - label_data) / batch_size   # shape: batch_size x num_class
                dloss_dWout = np.dot(h_rep.T, dloss_dlogits)              # shape: hidden_dim x num_class
                dloss_dbout = np.mean(dloss_dlogits, axis=0)

                # update weights
                momentum_W = 0.9 * momentum_W + dloss_dWout
                momentum_b = 0.9 * momentum_b + dloss_dbout
                Wout -= learning_rate*momentum_W
                bout -= learning_rate*momentum_b

                steps+=1
                if (steps % print_every == 0):
                    logits = np.dot(valid_rep, Wout) + bout
                    preds = np.argmax(logits, axis=1)
                    accuracy = np.mean(preds == np.argmax(valid_label, axis=1))
                    sys.stdout.write("\rEpoch ({:>4d}/{:4d})".format(i + 1, epochs)
                                     + "Step {:>6d} Loss {:>6.4f} Accuracy {:.2f}%".format(steps, loss, 100.0 * accuracy))

            if (i==0) or ((i+1) % nb_epochs == 0):
                print("\n")