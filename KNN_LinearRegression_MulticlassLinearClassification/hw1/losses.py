import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        idx = y.view(y.shape[0], 1)
        idx_expanded = idx.expand(*x_scores.shape)

        x_scores_for_y_class = torch.gather(x_scores, dim=1, index=idx_expanded)
        m = x_scores - x_scores_for_y_class + self.delta

        # Subtract self.delta because it is added in m[i][j] for j = y[i] for every i.
        m_positive = torch.clamp(m, min=0.0)
        sigma_with_yi_eq_j = torch.sum(m_positive)
        yi_eq_j = x_scores.shape[0]*self.delta
        loss = (sigma_with_yi_eq_j -  yi_eq_j) / float(x_scores.shape[0])
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx = {
            'x': x,
            'y': y,
            'm': m,
        }
        # ========================

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        m = self.grad_ctx['m']
        x = self.grad_ctx['x']
        y = self.grad_ctx['y']

        pos_m = torch.gt(m, torch.zeros(*m.shape))

        indices_of_true_class = y.view(m.shape[0], 1)

        ones_at_true_class = torch.zeros(*m.shape).scatter_(dim=1, index=indices_of_true_class,
                                                            src=torch.ones(*m.shape))
        
        ones_at_wrong_class = torch.ones(*m.shape) - ones_at_true_class.float()
        
        pos_ones_at_wrong_class = torch.mul(ones_at_wrong_class, pos_m.float())
        
        sums_for_true_classes = torch.sum(pos_ones_at_wrong_class, dim=1, keepdim=True)

        sum_of_indicators = torch.zeros(*m.shape).scatter_(dim=1, index=indices_of_true_class,
                                                           src=sums_for_true_classes)

        #coefficients_for_wrong_classes = torch.mul(ones_at_wrong_class, pos_m.float())

        g = pos_ones_at_wrong_class - sum_of_indicators

        grad = torch.mm(x.T, g) / float(m.shape[0])
        # ========================

        return grad
