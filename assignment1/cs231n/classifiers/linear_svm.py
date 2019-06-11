import numpy as np
from random import shuffle
from past.builtins import xrange

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
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  delta = 1

  # initialize the gradient as zero
  dW = np.zeros(W.shape)
  dS = np.zeros((num_train, num_classes))

  for i in xrange(num_train):
    # forward (affine layer)
    scores = X[i].dot(W)
    
    # loss function (svm hinge loss)
    correct_class_score = scores[y[i]]
    minus_cnt = 0
    for j in xrange(num_classes):  # iterate over all wrong classes
      if j == y[i]: # skip for the true class to only loop over incorrect classes
        continue
      margin = scores[j] - correct_class_score + delta # note delta = 1
      if margin > 0:
        loss += margin
        dS[i,j] = 1 / num_train  # SVM Loss gradient : max(0,1)
      else :
        minus_cnt += 1
  
    dS[i,y[i]] = -(num_classes-1-minus_cnt) / num_train
       
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
  dW = np.dot(X.T, dS) + reg * 2 * W # score gradient + regularization gradient

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  delta = 1

  # initialize the gradient as zero
  dW = np.zeros(W.shape)
  dS = np.zeros((num_train, num_classes))

  # forward (affine layer)
  scores = X.dot(W)
    
  # loss function (svm hinge loss)
  x_idx = np.arange(scores.shape[0]) # all the X index
  correct_class_score = scores[x_idx,y] # for all x, find y's score (correct class score)
  margin = np.maximum(0, (scores + delta) - correct_class_score[:, np.newaxis]) # max[0,sj - syi + delta]
  margin[x_idx,y] = 0 # set yi to 0 (becasue yi is 1)

  dS[margin > 0] = 1 / num_train # SVM Loss gradient : max(0,1)/N
  dS[x_idx,y] = -(np.count_nonzero(margin, axis=1))/ num_train # 정답의 경우 0이 아닌 Syi 개수/N
    
  loss = np.sum(margin)/ num_train
 
  # Add regularization to the loss.
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
  dW = np.dot(X.T, dS) + reg * 2 * W # score gradient + regularization gradient
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
