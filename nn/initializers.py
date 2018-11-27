import numpy as np
from scipy.stats import truncnorm  # For efficient truncated normal distribution


def truncated_normal(mean, std, shape, dtype='float32'):
    return truncnorm.rvs(mean, 2*std, size=shape).astype(dtype)


def variance_scaling_initializer(factor=2., mode='FAN_IN', uniform=False, dtype='float32'):
    assert mode in ['FAN_IN', 'FAN_OUT', 'FAN_AVG'], 'Mode {} was not registered {}'.format(mode, ['FAN_IN', 'FAN_OUT', 'FAN_AVG'])

    def _initializer(shape, dtype=dtype):
        if len(shape) == 2:  # Dense
            fan_in = float(shape[0])
            fan_out = float(shape[1])
        elif len(shape) == 4:  # Conv2D
            fan_in = float(shape[1])*shape[2]*shape[3]
            fan_out = float(shape[0])*shape[2]*shape[3]

        if mode == 'FAN_IN':
            n = fan_in
        elif mode == 'FAN_OUT':
            n = fan_out
        else:
            n = (fan_in + fan_out)/2.

        if uniform:
            limit = np.sqrt(3.0 * factor/n)
            return np.random.uniform(-limit, limit, shape).astype(dtype)
        else:
            std = np.sqrt(1.3*factor/n)
            return truncated_normal(0., std, shape, dtype=dtype)
    return _initializer


def xavier_initializer(uniform=True, dtype='float32'):
    return variance_scaling_initializer(factor=1., mode='FAN_AVG', uniform=uniform, dtype=dtype)


def zeros_initializer(dtype='float32'):
    def _initializer(shape, dtype=dtype):
        return np.zeros(shape).astype(dtype)
    return _initializer


def normal_initializer(mean=0., std=1e-3, dtype='float32'):
    def _initializer(shape, dtype=dtype):
        return np.random.normal(mean, std, shape).astype(dtype)
    return _initializer


def uniform_initializer(lower=-1, upper=1, dtype='float32'):
    def _initializer(shape, dtype=dtype):
        return np.random.uniform(lower, upper, shape).astype(dtype)
    return _initializer


def glorot_initializer(dtype='float32'):
    def _initializer(shape, dtype=dtype):
        limit = np.sqrt(6/np.sum(shape))
        return np.random.uniform(-limit, limit, shape).astype(dtype)
    return _initializer
