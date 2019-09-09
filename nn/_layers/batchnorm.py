import numpy as np
from .base import BaseLayer


class BatchNormalization(BaseLayer):
  def __init__(self, inp, momentum=0.99, epsilon=1e-5, scale=True, shift=True, name='batchnorm'):
    self.inp = inp
    self.momentum = momentum
    self.epsilon = epsilon
    self.shift = shift
    self.scale = scale
    self.name = name
    self.gamma = np.ones(inp.shape[1:])
    self.beta = np.zeros(inp.shape[1:])
    self.running_mean = np.zeros(inp.shape[1:]).astype('float32')
    self.running_var = np.zeros(inp.shape[1:]).astype('float32')

  def set_optimizer(self, optim):
    self.optim = optim
    if self.scale:
      self.optimizer_gamma = self.optim(self.gamma)
    if self.shift:
      self.optimizer_beta = self.optim(self.beta)
    if type(self.inp) != np.ndarray:
        self.inp.set_optimizer(optim)

  def get_weights(self):
    out = []
    if self.scale:
      out.append(self.gamma)
    if self.shift:
      out.append(self.beta)
    return tuple(out)

  def set_weights(self, gamma=None, beta=None):
    if gamma is not None and self.scale:
      self.gamma = gamma
    if beta is not None and self.shift:
      self.beta = beta

  def forward(self, x, is_training=False):
    if type(self.inp) != np.ndarray:
      x = self.inp.forward(x, is_training)

    if is_training:
      self.x = x
      self.mu = x.mean(axis=0)
      self.var = x.var(axis=0) + self.epsilon
      self.std = np.sqrt(self.var)
      self.z = (x - self.mu)/self.std
      out = self.gamma * self.z + self.beta

      self.running_mean = self.momentum*self.running_mean + (1-self.momentum)*self.mu
      self.running_var = self.momentum*self.running_var + (1-self.momentum)*self.var
    else:
      out = self.gamma*(x - self.running_mean)/np.sqrt(self.running_var + self.epsilon) + self.beta
    return out

  def compute_gradients(self, dout):
    self.gradients = []
    N = dout.shape[0]
    dfdz = dout*self.gamma
    dfdz_sum = np.sum(dfdz, axis=0)
    dx = dfdz - dfdz_sum/N - np.sum(dfdz*self.z, axis=0)*self.z/N
    dx /= self.std
    self.gradients.append(dx)
    if self.shift:
      dbeta = dout.sum(axis=0)
      self.gradients.append(dbeta)
    if self.scale:
      dgamma = np.sum(dout*self.z, axis=0)
      self.gradients.append(dgamma)

  def apply_gradients(self):
    if self.shift:
      self.beta, self.optimizer_beta = self.optim.update(self.beta, self.gradients[1], self.optimizer_beta)
    if self.scale:
      idx = 2 if self.shift else 1
      self.gamma, self.optimizer_gamma = self.optim.update(self.gamma, self.gradients[idx], self.optimizer_gamma)

  def minimize(self, dout):
    self.compute_gradients(dout)
    self.apply_gradients()
    if type(self.inp) != np.ndarray:
        self.inp.minimize(self.gradients[0])
    self.gradients = None
