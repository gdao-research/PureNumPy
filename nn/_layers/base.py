class BaseLayer(object):
    def set_optimizer(self, optim):
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError

    def set_weights(self, w, b=None):
        raise NotImplementedError

    def forward(self, x, is_training=False):
        raise NotImplementedError

    def compute_gradients(self, dout):
        raise NotImplementedError

    def apply_gradients(self):
        raise NotImplementedError

    def minimize(self, dout):
        raise NotImplementedError
