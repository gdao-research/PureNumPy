import numpy as np

class Dense:
    def __init__(self, inp, nb_filters, use_bias=True, kernel_initializer=None, bias_initializer=None, name='dense'):
        self.inp = inp
        self.name = name
        self.gradients = None
        self.use_bias = use_bias
        if kernel_initializer is None:
            self.w = np.random.normal(0, 1e-3, [np.prod(inp.shape[1:]), nb_filters]).astype('float32')
        else:
            self.w = kernel_initializer([np.prod(inp.shape[1:]), nb_filters], dtype='float32')
        if use_bias:
            if bias_initializer is None:
                self.b = np.zeros([nb_filters]).astype('float32')
            else:
                self.b = bias_initializer([nb_filters], 'float32')
        self.shape = (inp.shape[0], nb_filters)

    def set_optimizer(self, optim):
        self.optim = optim
        self.optimizer_w = self.optim(self.w)
        if self.use_bias:
            self.optimizer_b = self.optim(self.b, reg=0)
        if type(self.inp) != np.ndarray:
            self.inp.set_optimizer(self, optim)

    def get_weights(self):
        return (self.w, self.b) if self.use_bias else (self.w, )

    def set_weights(self, w, b=None):
        if b is None and self.use_bias:
            raise ValueError('Layer {self.name} use bias --> Please input b for set_weights')
        self.w = w
        if self.use_bias:    
            self.b = b

    def forward(self, x, is_training=False):
        if type(self.inp) != np.ndarray:
            x = self.inp.forward(x, is_training)
        out = np.dot(np.reshape(x, [self.x.shape[0], -1]), self.w)
        if self.use_bias:
            out += self.b
        if is_training:
            self.x = x
        return out

    def compute_gradients(self, dout):
        dx = np.reshape(np.dot(dout, self.w.T), self.x.shape)
        dw = np.dot(np.reshape(self.x, [self.x.shape[0], -1]).T, dout)
        db = np.sum(dout, axis=0)
        self.x = None
        self.gradients = (dx, dw, db)

    def apply_gradients(self):
        self.w, self.optimizer_w = self.optim.update(self.w, self.gradients[1], self.optimizer_w)
        if self.use_bias:
            self.b, self.optimizer_b = self.optim.update(self.b, self.gradients[2], self.optimizer_b)

    def minimize(self, dout):
        self.compute_gradients(dout)
        self.apply_gradients()
        if type(self.inp) != np.ndarray:
            self.inp.minimize(self.gradients[0])
        self.gradients = None

