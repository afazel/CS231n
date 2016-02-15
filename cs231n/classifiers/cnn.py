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
    # some computations to determine correct dimensions for weights and biases
    conv_s = 1
    conv_pad = (filter_size - 1) / 2
    pool_dim = 2
    pool_s = 2
    
    conv_hout = 1 + (input_dim[1] + 2*conv_pad - filter_size) / conv_s 
    conv_wout = 1 + (input_dim[2] + 2*conv_pad - filter_size) / conv_s
    pool_hout = 1 + (conv_hout - pool_dim) / pool_s
    pool_wout = 1 + (conv_wout - pool_dim) / pool_s
    D = num_filters * pool_hout * pool_wout
    
    self.params['W1'] = np.random.randn(num_filters, input_dim[0], filter_size, filter_size) * weight_scale
    self.params['b1'] = np.zeros([1, num_filters])
    
    self.params['W2'] = np.random.randn(D, hidden_dim) * weight_scale
    self.params['b2'] = np.zeros([1, hidden_dim]) 
    
    self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale
    self.params['b3'] = np.zeros([1, num_classes])
  
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    #a, conv_cache = conv_forward_naive(X, W1, b1, conv_param)
    a, conv_cache = conv_forward_fast(X, W1, b1, conv_param)
    b, relu1_cache = relu_forward(a)
    #c, pool_cache = max_pool_forward_naive(b, pool_param)
    c, pool_cache = max_pool_forward_fast(b, pool_param)
    
    # reshape c
    c_dim = c.shape
    N = c.shape[0]
    c = c.reshape([N, -1])
    
    d, aff1_cache = affine_forward(c, W2, b2)
    e, relu2_cache = relu_forward(d)
    scores, aff2_cache = affine_forward(e, W3, b3)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    # Compute loss
    loss, dscores = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))
        
    # Compute dW3
    de, dw3, db3 = affine_backward(dscores, aff2_cache)
    dd = relu_backward(de, relu2_cache)
    dc, dw2, db2 = affine_backward(dd, aff1_cache)
    
    # reshape dc
    dc = dc.reshape(c_dim)
    
    #db = max_pool_backward_naive(dc, pool_cache)
    db = max_pool_backward_fast(dc, pool_cache)
    da = relu_backward(db, relu1_cache)
    dx, dw1, db1 = conv_backward_fast(da, conv_cache)
    #dx, dw1, db1 = conv_backward_naive(da, conv_cache)
    
    #grads['X'] = dx
    grads['W1'] = dw1 + self.reg * W1
    grads['b1'] = db1
    grads['W2'] = dw2 + self.reg * W2
    grads['b2'] = db2
    grads['W3'] = dw3 + self.reg * W3
    grads['b3'] = db3
    
    for k, v in grads.iteritems():
      grads[k] = v.astype(self.dtype)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
