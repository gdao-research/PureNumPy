import numpy as np


class Conv2D:
    def __init__(self, inp, nb_filters, kernel_size=3, stride=1, padding=1, kernel_initializer=None, use_bias=True, bias_initializer=None, name='conv2d'):
        self.stride = stride
        self.padding = padding
        self.inp = inp
        self.name = name
        self.gradients = None
        self.use_bias = use_bias
        self.shape = (inp.shape[0], nb_filters, (inp.shape[2]+2*padding-kernel_size)//stride+1, (inp.shape[3]+2*padding-kernel_size)//stride+1)
        if kernel_initializer is None:
            self.w = np.random.normal(0, 1e-3, [nb_filters, inp.shape[1], kernel_size, kernel_size]).astype('float32')
        else:
            self.w = kernel_initializer((nb_filters, inp.shape[1], kernel_size, kernel_size), dtype='float32')
        if use_bias:
            if bias_initializer is None:
                self.b = np.zeros([nb_filters]).astype('float32')
            else:
                self.b = bias_initializer([nb_filters])

    def set_optimizer(self, optim):
        self.optim = optim
        self.optimizer_w = self.optim(self.w)
        if self.use_bias:
            self.optimizer_b = self.optim(self.b, reg=0)  # Does not apply regularizer for bias
        if type(self.inp) != np.ndarray:
            self.inp.set_optimizer(self, optim)

    def get_weights(self):
        return (self.w, self.b) if self.use_bias else (self.w, )

    def set_weights(self, w, b=None):
        if b is None and self.use_bias:
            raise ValueError(f'Layer {self.name} use bias --> Please input b for set_weights')
        self.w = w
        if self.use_bias:    
            self.b = b
        
    def forward(self, x, is_training=False):
        # _x = x if type(self.inp) == np.ndarray else self.inp.forward(x)
        if type(self.inp) != np.ndarray:
            x = self.inp.forward(x, is_training)
        N, C, H, W = self.x.shape
        F, _, HH, WW = self.w.shape
        x_padded = np.pad(self.x, ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)), mode='constant')
        H += 2*self.padding
        W += 2*self.padding
        out_h = (H-HH)//self.stride+1
        out_w = (W-WW)//self.stride+1
        strides = x.itemsize*np.array((H*W, W, 1, C*H*W, self.stride*W, self.stride))
        x_stride = np.lib.stride_tricks.as_strided(x_padded, shape=(C, HH, WW, N, out_h, out_w), strides=strides)
        x_cols = x_stride.reshape(C*HH*WW, N*out_h*out_w)
        out = self.w.reshape(F,-1).dot(x_cols)
        if self.use_bias:
            out += self.b.reshape((-1,1))
        out = out.reshape(F, N, out_h, out_w).transpose(1,0,2,3)
        if is_training:  # if training mode --> save cache
            self.x = x
            self.x_cols = x_cols
        return out

    def compute_gradients(self, dout):
        x_padded = np.pad(self.x, ((0,0), (0,0), (self.padding,self.padding), (self.padding, self.padding)), mode='constant')
        _, C, H, W = self.x.shape
        _, _, HH, WW = self.w.shape
        N, F, out_h, out_w = dout.shape
        db = np.sum(doub, axis=(0,2,3))
        dout_reshape = dout.transpose(1,0,2,3).reshape(F,-1)
        dw = dout_reshape.dot(self.x_cols.T).reshape(self.w.shape)
        dx_cols = (self.w.reshape(F,-1).T.dot(dout_reshape)).reshape((C, HH, WW, N, out_h, out_w))
        dx = np.zeros_like(x_padded)
        for h in range(out_h):
            for w in range(out_w):
                dx[:,:,self.stride*h:self.stride*h+HH,self.stride*w:self.stride*w+WW] += dx_cols[:,:,:,:,h,w].transpose(3,0,1,2)
        # Clear cache to save memory
        self.x = None
        self.x_cols = None
        self.gradients = (dx[:,:,self.padding:-self.padding,self.padding:-self.padding], dw, db)

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
