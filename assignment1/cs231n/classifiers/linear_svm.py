import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # (Added code per dW)
        # Following the derivative expression when calculating incorrect class of dW
        dW[:, j] += X[i]
        # Following the derivative expression when calculating correct class of dW
        dW[:, y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  # Following the derivative expression per W, averaging over the number of training data
  dW /= num_train
  # Following the derivative expression per W, adding the influence of regularization loss
  dW += 2 * reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  # Fully vectorized version
  # On Macbook Pro(early 2015), the time used by vetorized implementation is 
  # two degrees of magnitude less than that by naive implementation.
  loss_matrix = np.dot(X, W)
  real_score = loss_matrix[np.arange(num_train), y].reshape(-1, 1)
  loss_criterior = np.maximum(loss_matrix - real_score + 1.0, np.zeros((num_train, num_class)))
  loss_criterior[np.arange(num_train), y] = 0
  loss = np.sum(loss_criterior) / num_train
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # Semi-vectorized version
  # On Macbook Pro(early 2015), the time used by vetorized implementation is 
  # about half of that by naive implementation. Not much difference.
  #for i in range(num_train):
  #  dW += np.dot(X[i:i+1, :].T, (loss_criterior > 0)[i:i+1, :])
  #  dW[:, y[i]] -= np.sum(dW, axis=1)
  
  # Fully vectorized version
  # On Macbook Pro(early 2015), the time used by vetorized implementation is 
  # two degrees of magnitude less than that by naive implementation.
  loss_criterior_bi = np.zeros(loss_criterior.shape)
  loss_criterior_bi[loss_criterior > 0] = 1
  loss_criterior_bi[np.arange(num_train), y] = -np.sum(loss_criterior_bi, axis=1)
  dW = np.dot(X.T, loss_criterior_bi)

  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
