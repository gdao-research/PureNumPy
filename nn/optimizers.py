import numpy as np 


class SGD:
    def __init__(self, lr=1e-2):
        self.lr = np.asarray(lr, dtype='float32')
        self.t = np.asarray(1, dtype='float32')

    def __call__(self, reg=0.):
        self.reg = np.asarray(reg, dtype='float32')
        return (None, )

    def increase_t(self):
        self.t += 1

    def set_lr(self, lr, dtype='float32'):
        self.lr = np.asarray(lr, dtype=dtype)

    def get_train_step(self):
        return self.t

    def update(self, x, dx, params=(None,)):
        dx += self.reg*x
        next_x = x - self.lr*dx
        return next_x, params


class SGDMomentum:
    def __init__(self, lr=1e-2, momentum=0.9):
        self.lr = np.asarray(lr, dtype='float32')
        self.momentum = np.asarray(momentum, dtype='float32')
        self.t = np.asarray(1, dtype='float32')

    def __call__(self, x, reg=0., dtype='float32'):
        self.reg = np.asarray(reg, dtype='float32')
        return (np.zeros_like(x, dtype=dtype), )

    def increase_t(self):
        self.t += 1
    
    def set_lr(self, lr, dtype='float32'):
        self.lr = np.asarray(lr, dtype=dtype)

    def get_train_step(self):
        return self.t

    def update(self, x, dx, params):
        dx += self.reg*x
        params = self.momentum*params[0] - self.lr*dx
        next_x = x + params
        return next_x, (params,)


class RMSProp:
    def __init__(self, lr=1e-2, decay_rate=0.99, epsilon=1e-8):
        self.lr = np.asarray(lr, dtype='float32')
        self.decay_rate = np.asarray(decay_rate, dtype='float32')
        self.epsilon = np.asarray(epsilon, dtype='float32')
        self.t = np.asarray(1, dtype='float32')

    def __call__(self, x, reg=0., dtype='float32'):
        self.reg = np.asarray(reg, dtype='float32')
        return (np.zeros_like(x, dtype=dtype), )

    def increase_t(self):
        self.t += 1

    def set_lr(self, lr, dtype='float32'):
        self.lr = np.asarray(lr, dtype=dtype)

    def get_train_step(self):
        return self.t

    def update(self, x, dx, params):
        dx += self.reg*x
        params = self.decay_rate*params[0] + (1-self.decay_rate)*dx**2
        next_x = x - self.lr*dx/(np.sqrt(params) + self.epsilon)
        return next_x, (params,)


class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = np.asarray(lr, dtype='float32')
        self.beta1 = np.asarray(beta1, dtype='float32')
        self.beta2 = np.asarray(beta2, dtype='float32')
        self.epsilon = np.asarray(epsilon, dtype='float32')
        self.t = np.asarray(1, dtype='float32')

    def __call__(self, x, reg=0., dtype='float32'):
        self.reg = np.asarray(reg, dtype='float32')
        return (np.zeros_like(x, dtype=dtype), np.zeros_like(x, dtype=dtype))

    def increase_t(self):
        self.t += 1

    def set_lr(self, lr, dtype='float32'):
        self.lr = np.asarray(lr, dtype=dtype)

    def get_train_step(self):
        return self.t

    def update(self, x, dx, params):
        dx += self.reg*x
        params0 = self.beta1*params[0] + (1 - self.beta1)*dx
        params1 = self.beta2*params[1] + (1 - self.beta2)*(dx**2)
        learning_rate = self.lr*np.sqrt(1-self.beta2**self.t)/(1-self.beta1**self.t)
        next_x = x - learning_rate*params0/(np.sqrt(params1) + self.epsilon)
        return next_x, (params0, params1)
