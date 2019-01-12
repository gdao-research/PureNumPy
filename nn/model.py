class BaseModel(object):
    def __init__(self, inp, outp, loss_fn, optim, name='model'):
        self.name = name
        self.loss_fn = loss_fn
        self.model = self.predictor(inp, outp)
        self.optim = self.set_optimizer(optim)

    def predictor(self, inp, outp):
        pass

    def set_optimizer(self, optim):
        self.model[-1].set_optimizer(optim)
        return optim

    def forward(self, x, y=None, is_training=False):
        out = self.model[-1].forward(x, is_training)
        return self.loss_fn(out) if y is None else self.loss_fn(out, y)

    def fit(self, x, y):
        loss_fn, dout = self.forward(x, y, is_training=True)
        self.model[-1].minimize(dout[0])
        self.optim.increase_t()
        return loss_fn
