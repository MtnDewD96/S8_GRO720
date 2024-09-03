import numpy as np
from dnn_framework.layer import Layer

class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """

    def __init__(self, input_count, output_count):
        super().__init__()
        self.W = np.random.normal(0,2/(input_count+output_count),(output_count, input_count))
        self.B = np.random.normal(0,2/output_count,output_count)

    def get_parameters(self):
        return {
            'w': self.W,
            'b': self.B
        }

    def get_buffers(self):
        return {}

    def forward(self, x):
        output = x @ self.W.T + self.B
        return output, x

    def backward(self, output_grad, cache):
        input_grad = output_grad @ self.W
        w_grad = output_grad.T @ cache
        b_grad = np.sum(output_grad, axis=0)

        param_dict = {
            'w': w_grad,
            'b': b_grad
        }

        return input_grad, param_dict


class BatchNormalization(Layer):
    """
    This class implements a batch normalization layer.
    """

    def __init__(self, input_count, alpha=0.1):
        super().__init__()
        self.input = np.zeros(input_count)
        self.global_mean = np.zeros(input_count)
        self.global_variance = np.ones(input_count)
        self.gamma = np.ones(input_count)
        self.beta = np.zeros(input_count)
        self.alpha = alpha


    def get_parameters(self):
        return {
            'gamma': self.gamma,
            'beta': self.beta
        }

    def get_buffers(self):
        return {
            'global_mean': self.global_mean,
            'global_variance': self.global_variance
        }

    def forward(self, x):
        if self.is_training():
            return self._forward_training(x)
        return self._forward_evaluation(x)

    def _forward_training(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_variance = np.var(x, axis=0)
        batch_mean_diff = x-batch_mean
        batch_std = np.sqrt(batch_variance + 1e-10)
        x_pred = batch_mean_diff/batch_std
        y = x_pred*self.gamma + self.beta

        self.global_mean = (1-self.alpha)*self.global_mean + self.alpha*batch_mean
        self.global_variance = (1-self.alpha)*self.global_variance + self.alpha*batch_variance
        return y, (x_pred, batch_mean_diff, batch_std)

    def _forward_evaluation(self, x):
        x_pred = (x-self.global_mean)/np.sqrt(self.global_variance+1e-10)
        y = x_pred*self.gamma + self.beta
        return y, None

    def backward(self, output_grad, cache):
        x_pred, batch_mean_diff, batch_std = cache

        gamma_grad = np.sum(output_grad * x_pred, axis=0)
        beta_grad = np.sum(output_grad, axis=0)

        param_dict = {
            'gamma': gamma_grad,
            'beta': beta_grad
        }

        d_xpred = output_grad*self.gamma
        d_mean = -np.sum(d_xpred/batch_std, axis=0)
        d_variance = -np.sum(d_xpred*batch_mean_diff/(batch_std**3), axis=0)/2
        dx = d_xpred/batch_std + (2*d_variance*batch_mean_diff + d_mean)/x_pred.shape[0]

        return dx, param_dict


class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """

    def get_parameters(self):
        return {}

    def get_buffers(self):
        return {}

    def forward(self, x):
        output =  1/(1+np.exp(-x))
        return output, output

    def backward(self, output_grad, cache):
        return output_grad*cache*(1 - cache), {}


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """

    def get_parameters(self):
        return {}

    def get_buffers(self):
        return {}

    def forward(self, x):
        output = np.clip(x, 0, None)
        return output, output

    def backward(self, output_grad, cache):
        return output_grad*np.where(cache>0, 1, 0), {}
