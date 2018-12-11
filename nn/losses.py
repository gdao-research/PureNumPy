import numpy as np 


def softmax(logit, y=None):
    N = logit.shape[0]
    norm = logit - np.max(logit, axis=1, keepdims=True)
    Z = np.sum(np.exp(norm), axis=1, keepdims=True)
    log_probs = norm - np.log(Z)
    probs = np.exp(log_probs)
    if y is None:
        return probs

    if y.ndim == 1:  # Raw categories
        loss = -np.mean(log_probs[np.arange(N), y])
        probs[np.arange(N), y] -= 1
    else:  # One-hot encoded categories
        loss = -np.mean(log_probs[y.astype('bool')])
        probs[y.astype('bool')] -= 1
    return loss, (probs/N, )


def mean_squared_error(logit, y=None):
    # J = (y-yHat)**2
    N = logit.shape[0]
    if y is None:
        return logit
    dx = logit - y
    squared_error = dx**2
    loss = squared_error.mean()
    return loss, (2*dx/N, )
