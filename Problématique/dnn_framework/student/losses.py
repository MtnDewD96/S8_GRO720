import numpy as np

from dnn_framework.loss import Loss


class CrossEntropyLoss(Loss):
    """
    This class combines a softmax activation function and a cross entropy loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: (N, C))
        :param target: The target classes (shape: (N,))
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """

        # Softmax of every input to get probability distribution + log probability
        s = softmax(x)
        log_prob = np.log(s + 1e-5)
        # We only keep the probabilities of the non-zero targets (we find the ones in the one-hot encoding)
        # This arises from the cross entropy formula y_j*log(yhat_j), where y_j will be 1 for a single class and 0 for the rest
        target_arange = np.arange(len(target))
        target_log_prob = log_prob[target_arange, target]
        # Get negative mean of all elements in batch
        loss = -np.mean(target_log_prob)

        # Create one-hot vector for target classes
        oh_target = np.zeros_like(x)
        oh_target[target_arange, target] = 1

        # Compute gradient using softmax
        grad_loss = (s - oh_target)/s.shape[0]

        return loss, grad_loss


def softmax(x):
    """
    :param x: The input tensor (shape: (N, C))
    :return The softmax of x
    """

    # Max value to avoid numerical instability
    mx = np.max(x,axis=1,keepdims=True)
    e_x = np.exp(x-mx)
    return e_x/np.sum(e_x,axis=1, keepdims=True)


class MeanSquaredErrorLoss(Loss):
    """
    This class implements a mean squared error loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: any)
        :param target: The target tensor (shape: same as x)
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        predict_diff = x - target
        loss = np.mean(predict_diff**2)
        grad_loss = 2*predict_diff/x.size
        return loss, grad_loss

