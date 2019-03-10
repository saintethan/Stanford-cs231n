import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  
  for i in range(num_train):
    score = np.dot(X[i], W)
    # Normalization trick to avoid numerical instability, refer to 
    # http://cs231n.github.io/linear-classify/#softmax
    score -= np.amax(score)
    denominator = np.sum(np.exp(score))
    exp_correct_score = np.exp(score[y[i]])
    loss += -np.log(exp_correct_score / denominator)
    for j in range(num_class):
      dW[:, j] += X[i] * ((np.exp(score[j]) / denominator) - (j == y[i]))
  
  loss /= num_train
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]

  score = np.dot(X, W)
  score -= np.amax(score, axis=1).reshape(num_train, -1)

  denominator = np.sum(np.exp(score), axis=1)
  numerator = np.exp(score[np.arange(num_train), y])
  loss += np.sum(-np.log(numerator / denominator))
  loss /= num_train
  loss += reg * np.sum(W * W)

  score_trans = np.zeros_like(score)
  score_trans += (np.exp(score) / denominator.reshape(num_train, -1))
  score_trans[np.arange(num_train), y] -= 1
  dW = np.dot(X.T, score_trans)
  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

