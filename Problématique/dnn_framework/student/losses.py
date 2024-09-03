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

        # Could be optimized with softmax exponential being cancelled by log
        s = softmax(x)
        target_arange = np.arange(len(target))
        log_prob = np.log(s)
        target_log_prob = log_prob[target_arange, target]
        loss = -np.mean(target_log_prob)

        oh_target = np.zeros_like(x)
        oh_target[target_arange, target] = 1

        # Not the equation from the student guide?!?! Why?!?! should be -(target/s)
        grad_loss = (s - oh_target) /s.shape[0]
        return loss, grad_loss


def softmax(x):
    """
    :param x: The input tensor (shape: (N, C))
    :return The softmax of x
    """
    e_x = np.exp(x)
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

