import numpy as np
from random import shuffle
from past.builtins import xrange
from cs231n.classifiers.functions import softmax
from cs231n.classifiers.functions import cross_entropy

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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  # initialize the gradient as zero
  dW = np.zeros_like(W)
  dS = np.zeros((num_train, num_classes))

  for i in xrange(num_train):
    # forward
    scores = X[i].dot(W) # affine
    y_pred = softmax(scores) # softmax (vector return)
    t = np.zeros(num_classes) # one hot encoding
    t[y[i]] = 1
   
    # loss function
    loss += cross_entropy(y_pred, t) # cross entropy (scalar return)
    #print(cross_entropy(y_pred, t))
    
    # backward
    dS[i] = (y_pred - t) / num_train  # softmax gradient

  loss /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  dW = np.dot(X.T, dS) + reg * 2 * W # score gradient + regularization gradient

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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  # initialize the gradient as zero
  dW = np.zeros_like(W) # affine
  dS = np.zeros((num_train, num_classes))
  t = np.zeros((num_train, num_classes)) # one hot encoding
  t[np.arange(num_train), y] = 1 # one hot encoding
  
  # forward
  scores = X.dot(W)
  y_pred = softmax(scores) # softmax

  # loss function
  loss += cross_entropy(y_pred, t) # cross entropy

  # backward
  dS = (y_pred - t) / num_train # softmax gradient

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
    
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  dW = np.dot(X.T, dS) + reg * 2 * W # score gradient + regularization gradient
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
