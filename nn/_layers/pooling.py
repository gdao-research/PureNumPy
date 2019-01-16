import numpy as np
from .base import BaseLayer


class MaxPooling2D(BaseLayer):
    def __init__(self, inp, kernel=2, stride=2, name='maxpooling2d'):
        self.inp = inp
        self.kernel = kernel
        self.stride = stride
        self.gradients = None
        self.name = name
        self.N, self.C, self.H, self.W = self.inp.shape
        self.out_H = np.int(((self.H - kernel)/stride) + 1)
        self.out_W = np.int(((self.W - kernel)/stride) + 1)
        self.shape = (self.N, self.C, self.out_H, self.out_W)
        # Recommend to get fast computation because naive is quite slow
        self.method = 'fast' if kernel == stride and self.H % kernel == 0 and self.W % kernel == 0 else 'naive'

    def set_optimizer(self, optim=None):
        if type(self.inp) != np.ndarray:
            self.inp.set_optimizer(optim)

    def get_weights(self):
        return (None, )

    def set_weights(self, w=None, b=None):
        pass

    def forward(self, x, is_training=False):
        def forward_fast(self, x, is_training=False):
            if type(self.inp) != np.ndarray:
                x = self.inp.forward(x, is_training)
            x_reshaped = x.reshape(self.N, self.C, self.H//self.kernel, self.kernel, self.W//self.kernel, self.kernel)
            out = x_reshaped.max(axis=3).max(axis=4)
            if is_training:
                self.x = x
                self.x_reshaped = x_reshaped
                self.out = out
            return out

        def forward_naive(self, x, is_training=False):
            if type(self.inp) != np.ndarray:
                x = self.inp.forward(x)
            out = np.zeros([self.N, self.C, self.out_H, self.outW])
            for n in range(self.N):
                for c in range(self.C):
                    for h in range(self.out_H):
                        for w in range(self.out_W):
                            out[n, c, h, w] = np.max(x[n, c, 
                                                       h*self.stride:h*self.stride+self.kernel, 
                                                       w*self.stride:w*self.stride+self.kernel])
            if is_training:
                self.x = x
            return out

        return forward_fast(x, is_training) if self.method == 'fast' else forward_naive(x, is_training)

    def compute_gradients(self, dout):
        def compute_gradients_fast(dout):
            dx_reshaped = np.zeros_like(self.x_reshaped)
            out_newaxis = self.out[:, :, :, np.newaxis, :, np.newaxis]
            mask = (self.x_reshaped == out_newaxis)
            dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
            dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
            dx_reshaped[mask] = dout_broadcast[mask]
            dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
            dx = dx_reshaped.reshape(self.x.shape)
            self.x = None
            self.x_reshaped = None
            self.out = None
            return (dx, )

        def compute_gradients_naive(dout):
            dx = np.zeros_like(self.x)
            _, _, dout_H, dout_W = dout.shape
            for n in range(self.N):
                for c in range(self.C):
                    for h in range(dout_H):
                        for w in range(dout_W):
                            max_index = np.argmax(self.x[n, c, h*self.stride:h*self.stride+self.kernel, w*self.stride:w*self.stride+self.kernel])
                            max_coord = np.unravel_index(max_index, [self.kernel, self.kernel])
                            dx[n, c, h*self.stride:h*self.stride+self.kernel, w*self.stride:w*self.stride+self.kernel][max_coord] = dout[n,c,h,w]
            self.x = None
            return (dx, )

        self.gradients = compute_gradients_fast(dout) if self.method == 'fast' else compute_gradients_naive(dout)

    def apply_gradients(self):
        pass

    def minimize(self, dout):
        self.compute_gradients(dout)
        # self.apply_gradients()
        if type(self.inp) != np.ndarray:
            self.inp.minimize(self.gradients[0])
        self.gradients = None
