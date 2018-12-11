import numpy as np 


class SGD:
    def __init__(self, lr=1e-2, reg=0.):
        self.lr = np.asarray(lr, dtype='float32')
        self.reg = np.asarray(reg, dtype='float32')

    def __call__(self):
        pass

    def update(self, x, dx):
        dx += self.reg*x
        next_x = x - self.lr*dx
        return next_x


class SGDMomentum:
    def __init__(self, lr=1e-2, momentum=0.9, reg=0.):
        self.lr = np.asarray(lr, dtype='float32')
        self.momentum = np.asarray(momentum, dtype='float32')
        self.reg = np.asarray(reg, dtype='float32')
        self.v = None

    def __call__(self, x, dtype='float32'):
        self.v = np.zeros_like(x, dtype=dtype)

    def update(self, x, dx):
        dx += self.reg*x
        self.v = self.momentum*self.v - self.lr*dx
        next_x = x + self.v
        return next_x


class RMSProp:
    def __init__(self, lr=1e-2, decay_rate=0.99, epsilon=1e-8, reg=0.):
        self.lr = np.asarray(lr, dtype='float32')
        self.decay_rate = np.asarray(decay_rate, dtype='float32')
        self.epsilon = np.asarray(epsilon, dtype='float32')
        self.reg = np.asarray(reg, dtype='float32')
        self.cache = None

    def __call__(self, x, dtype='float32'):
        self.cache = np.zeros_like(x, dtype=dtype)

    def update(self, x, dx):
        dx += self.reg*x
        self.cache = self.decay_rate*self.cache + (1-self.decay_rate)*dx**2
        next_x = x - self.lr*dx/(np.sqrt(self.cache) + self.epsilon)
        return next_x


class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, reg=0.):
        self.lr = np.asarray(lr, dtype='float32')
        self.beta1 = np.asarray(beta1, dtype='float32')
        self.beta2 = np.asarray(beta2, dtype='float32')
        self.epsilon = np.asarray(epsilon, dtype='float32')
        self.reg = np.asarray(reg, dtype='float32')
        self.t = np.asarray(0, dtype='float32')
        self.m = None
        self.v = None

    def __call__(self, x, dtype='float32'):
        self.m = np.zeros_like(x, dtype=dtype)
        self.v = np.zeros_like(x, dtype=dtype)

    def update(self, x, dx):
        self.t += 1
        dx += self.reg*x
        self.m = self.beta1*self.m + (1 - self.beta1)*dx
        self.v = self.beta2*self.v + (1 - self.beta2)*(dx**2)
        learning_rate = self.lr*np.sqrt(1-self.beta2**self.t)/(1-self.beta1**self.t)
        next_x = x - learning_rate*self.m/(np.sqrt(self.v) + self.epsilon)
        return next_x
