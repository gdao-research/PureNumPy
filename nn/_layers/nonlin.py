import numpy as np

class ReLU:
    def __init__(self, inp, name='relu'):
        self.shape = inp.shape
        self.name = name
        self.inp = inp
        self.gradients = None

    def set_optimizer(self, optim=None):
        if type(self.inp) != np.ndarray:
            self.inp.set_optimizer(self, optim)

    def get_weights(self):
        return (None, )
    
    def set_weights(self, w=None, b=None):
        pass

    def forward(self, x, is_training=False):
        if type(self.inp) != np.ndarray:
            x = self.inp.forward(x, is_training)
        out = x.copy()
        out[out<0] = 0
        if is_training:
            self.x = x
        return out

    def compute_gradients(self, dout):
        dx = dout * (self.x >= 0)
        self.gradients = (dx, )

    def apply_gradients(self):
        pass

    def minimize(self, dout):
        self.compute_gradients(dout)
        # self.apply_gradients()
        if type(self.inp) != np.ndarray:
            self.inp.minimize(self.gradients[0])
        self.gradients = None


class PReLU:
    def __init__(self, inp, alpha=0.001, name='prelu'):
        self.alpha = np.asarray(alpha, dtype='float32')
        self.shape = inp.shape
        self.name = name
        self.inp = inp
        self.gradients = None

    def set_optimizer(self, optim):
        self.optim = optim
        self.optimizer_alpha = self.optim(self.alpha)
        if type(self.inp) != np.ndarray:
            self.inp.set_optimizer(optim)

    def get_weights(self):
        return (self.alpha, )

    def set_weights(self, alpha):
        self.alpha = alpha

    def forward(self, x, is_training=False):
        if type(self.inp) != np.ndarray:
            x = self.inp.forward(x, is_training)
        out = x.copy()
        out[out<0] = out[out<0]*self.alpha
        if is_training:
            self.x = x
        return out

    def compute_gradients(self, dout):
        dx = dout*(self.x >= 0)
        dx[self.x<0] = dout[self.x<0] * self.alpha
        dalpha = np.sum(dout*(self.x<0)*self.x)
        self.x = None
        self.gradients = (dx, dalpha)

    def apply_gradients(self):
        self.alpha, self.optimizer_alpha = self.optim.update(self.alpha, self.gradients[1], self.optimizer_alpha)

    def minimize(self, dout):
        self.compute_gradients(dout)
        self.apply_gradients()
        if type(self.inp) != np.ndarray:
            self.inp.minimize(self.gradients[0])
        self.gradients = None
