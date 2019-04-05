import torch

class EMA():
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}
        
    def register_model(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.register(name, param.data)

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def update_model(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param = self.update(name, param)

    def update(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average

class DelayedWeight(object):
    def __init__(self, params, src_params):

        self.params = list(params)
        self.src_params = list(src_params)

        for p, src_p in zip(self.params, self.src_params):
            p.data[:] = src_p.data[:]

    def step(self):
        for p, src_p in zip(self.params, self.src_params):
            p.data.set_(src_p.data)

    def zero_grad(self):
        pass

class WeightEMA (object):
    def __init__(self, params, src_params, alpha=0.998):

        self.params = list(params)
        self.src_params = list(src_params)
        self.alpha = alpha

        for p, src_p in zip(self.params, self.src_params):
            p.data[:] = src_p.data[:]

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for p, src_p in zip(self.params, self.src_params):
            p.data.mul_(self.alpha)
            p.data.add_(src_p.data * one_minus_alpha)

    def zero_grad(self):
        pass
