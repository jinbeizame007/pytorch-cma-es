import numpy as np


# https://github.com/openai/evolution-strategies-starter/blob/951f19986921135739633fb23e55b2075f66c2e6/es_distributed/es.py#L25

class RunningStat(object):
    def __init__(self, shape, eps):
        self.sum = np.zeros(shape, dtype=np.float32)
        self.sumsq = np.full(shape, eps, dtype=np.float32)
        self.count = eps

    def increment(self, s, ssq, c):
        self.sum += s
        self.sumsq += ssq
        self.count += c

    @property
    def mean(self):
        return self.sum / self.count

    @property
    def std(self):
        return np.sqrt(np.maximum(self.sumsq / self.count - np.square(self.mean), 1e-2))

    def set_from_init(self, init_mean, init_std, init_count):
        self.sum[:] = init_mean * init_count
        self.sumsq[:] = (np.square(init_mean) + np.square(init_std)) * init_count
        self.count = init_count


def compute_centered_ranks(x):
    ranks = np.empty(x.shape[0], dtype=np.float32)
    ranks[x.argsort()] = np.linspace(-0.5, 0.5, x.shape[0], dtype=np.float32)
    return ranks

def compute_weight_decay(weight_decay, model_param_list):   
    model_param_grid = np.array(model_param_list)
    return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)