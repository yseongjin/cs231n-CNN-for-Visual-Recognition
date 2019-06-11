from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################      
        # pass conv_param to the forward pass for the convolutional layer
        self.conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        # pass pool_param to the forward pass for the max-pooling layer
        self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        C, H, W = input_dim
        FN, FH, FW = num_filters, filter_size, filter_size
        stride, pad = self.conv_param['stride'], self.conv_param['pad']
        OH1, OW1 = int(1+(H + 2*pad - FH)/stride), int(1+(W + 2*pad - FW)/stride)
        
        FH, FW = self.pool_param['pool_height'], self.pool_param['pool_width']
        stride, pad = self.pool_param['stride'], 0
        OH2, OW2 = int(1+(OH1 + 2*pad - FH)/stride), int(1+(OW1 + 2*pad - FW)/stride)
    
        fc_input_dim = FN*OH2*OW2
        
        self.params['W1'] = np.random.randn(num_filters, C, filter_size, filter_size)*weight_scale
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = np.random.randn(fc_input_dim, hidden_dim)*weight_scale
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = np.random.randn(hidden_dim, num_classes)*weight_scale
        self.params['b3'] = np.zeros(num_classes)

        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        scores = None
        caches = {}
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        N, C, H, W = X.shape
        a, cache = conv_relu_pool_forward(X, W1, b1, self.conv_param, self.pool_param)
        caches['cache1'] = cache
        
        final_conv_shape = a.shape
        a = a.reshape(N, -1) # Fully Connected Layer의 Input 형태로 변형
        a, cache = affine_relu_forward(a, W2, b2)
        caches['cache2'] = cache
        
        a, cache = affine_forward(a, W3, b3)
        caches['cache3'] = cache
        
        scores = a
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, da = softmax_loss(scores, y)
        loss += (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))*self.reg / 2 # L2 Loss
        
        cache = caches['cache3']
        da, dW, db = affine_backward(da, cache)
        dW += self.reg * W3 # L2 regularization term gradient
        grads['W3'], grads['b3']= dW, db
        
        cache = caches['cache2']
        da, dW, db = affine_relu_backward(da, cache)
        dW += self.reg * W2 # L2 regularization term gradient
        grads['W2'], grads['b2']= dW, db
        
        da = da.reshape(final_conv_shape) # 최종 Convolution Layer 모양으로 복귀
        cache = caches['cache1']
        dx, dW, db = conv_relu_pool_backward(da, cache)
        dW += self.reg * W1 # L2 regularization term gradient
        grads['W1'], grads['b1']= dW, db
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
