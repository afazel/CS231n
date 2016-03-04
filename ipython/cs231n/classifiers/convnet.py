import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class MyFullyConvNet(object):
  """
  A full convolutional network with the following architecture:
  
  [conv-relu-pool]x(convlayer_num) - [affine-relu]x(afflayer_num) - affine - [softmax]
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, convlayer_params, affinelayer_params, weight_scale=1e-3, 
               reg=0.0, input_dim=(1, 48, 48), num_classes=7, 
               dtype=np.float32):              
    """
    Initialize a new network.
    
    Inputs:
    - convlayer_params: a dictionary that contains the properties of the convolution 
      layer for [conv-relu-pool] including:
      number of conv layers, num_filters, filter_size, stride, pad, spatial_batch_norm,
      pool_dim, pool_stride
    - affinelayer_params: a dictionary that contains the properties of the affine
      layer of the form [affine-relu] including:
      number pf affine layers, hidden_dim, batch_norm (True/False), dropout(True/False)
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights. 
    - reg: Scalar giving L2 regularization strength
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_classes: Number of scores to produce from the final affine layer.    
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    C_in, H_in, W_in = input_dim
    
    self.conv_num = convlayer_params['convlayer_num']
    num_filters = convlayer_params['num_filters']
    filter_size = convlayer_params['filter_size']
    use_sbatch = convlayer_params['s_batch_norm']
    conv_s = convlayer_params['stride']
    conv_pad = convlayer_params['pad']
    pool_dim = convlayer_params['pool_dim']
    pool_s = convlayer_params['pool_stride']
    
    # conv_param required for the forward pass for the convolutional layer
    self.conv_param = {'stride': conv_s, 'pad': conv_pad}

    # pool_param required for the forward pass for the max-pooling layer
    self.pool_param = {'pool_height': pool_dim, 'pool_width': pool_dim, 'stride': pool_s}
    
    self.aff_num = affinelayer_params['afflayer_num']
    use_batch = affinelayer_params['batch_norm']
    use_dropout = affinelayer_params['dropout']
    hidden_dim = affinelayer_params['hidden_dim']
    
    # Initialize convolution layers
    C = C_in
    H = H_in
    W = H_in
    for i in range(self.conv_num):
        
        self.params['conv_W'+str(i+1)] = np.random.randn(num_filters[i], C, filter_size, filter_size) * weight_scale
        self.params['conv_b'+str(i+1)] = np.zeros(num_filters[i])
        C = num_filters[i]
        
        # determine correct dimensions after one conv 
        conv_hout = 1 + (H + 2*conv_pad - filter_size) / conv_s 
        conv_wout = 1 + (W + 2*conv_pad - filter_size) / conv_s
        
        # determine correct dimensions after one pool
        H = 1 + (conv_hout - pool_dim) / pool_s
        W = 1 + (conv_wout - pool_dim) / pool_s
              
    # Initialize affine layers   
    first_dim = num_filters[-1] * H * W
    for i in range(self.aff_num):
    
        self.params['aff_W'+str(i+1)] = np.random.randn(first_dim, hidden_dim[i]) * weight_scale
        self.params['aff_b'+str(i+1)] = np.zeros([1,hidden_dim[i]]) 
        first_dim = hidden_dim[i]
        
    self.params['aff_W'+str(self.aff_num+1)] = np.random.randn(first_dim, num_classes) * weight_scale
    self.params['aff_b'+str(self.aff_num+1)] = np.zeros([1, num_classes]) 
     
    ############################################################################
   
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the full convolutional network.
    
    """
    
    X = X.astype(self.dtype)
    
    scores = None
    ############################################################################
    # Implement the forward pass for the full convolutional net,                   
    conv_cache = {}
    reluconv_cache = {}
    aff_cache = {}
    reluaff_cache = {}
    pool_cache = {}
    
    
    # Implement forward pass for convolutional layers
    x_input = X
    for i in range(1, self.conv_num+1):
    
        W, b = self.params['conv_W'+str(i)] , self.params['conv_b'+str(i)]
    
        #conv_out, conv_cache = conv_forward_naive(X, W1, b1, conv_param)
        conv_out, conv_cache['conv'+str(i)] = conv_forward_fast(x_input, W, b, self.conv_param)
    
        relu_out, reluconv_cache['conv'+str(i)] = relu_forward(conv_out)
    
        #pool_out, pool_cache = max_pool_forward_naive(b, pool_param)
        pool_out, pool_cache['conv'+str(i)] = max_pool_forward_fast(relu_out, self.pool_param)
        
        x_input = pool_out
    
    # Implement forward pass for affine layers
    # need to reshape x_input
    original_dim = pool_out.shape
    x_input = x_input.reshape([x_input.shape[0], -1])
    
    for i in range(1, self.aff_num+1):
    
        W, b = self.params['aff_W'+str(i)] , self.params['aff_b'+str(i)]
        
        aff_out, aff_cache['aff'+str(i)] = affine_forward(x_input, W, b)
        relu_out, reluaff_cache['aff'+str(i)] = relu_forward(aff_out)
        x_input = relu_out
    
    # AND finally compute the scores
    W, b = self.params['aff_W'+str(self.aff_num+1)] , self.params['aff_b'+str(self.aff_num+1)]
    scores, aff_cache['aff'+str(self.aff_num+1)] = affine_forward(x_input, W, b)
    
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # Implement the backward pass for the full convolutional net:
    
    # Compute loss
    loss, dscores = softmax_loss(scores, y)
    
    for i in range(1, self.conv_num+1):
        loss += 0.5 * self.reg * np.sum(self.params['conv_W'+str(i)]**2)
        
    for i in range(1, self.aff_num+2):
        loss += 0.5 * self.reg * np.sum(self.params['aff_W'+str(i)]**2)
        
    # Compute dW and db right before the loss computation
    dout, grads['aff_W'+str(self.aff_num+1)], grads['aff_b'+str(self.aff_num+1)] = affine_backward(dscores, aff_cache['aff'+str(self.aff_num+1)])
    grads['aff_W'+str(self.aff_num+1)] += self.reg * self.params['aff_W'+str(self.aff_num+1)]
    
    for i in range(self.aff_num, 0, -1):
    
        dout = relu_backward(dout, reluaff_cache['aff'+str(i)])
        dout, grads['aff_W'+str(i)], grads['aff_b'+str(i)] = affine_backward(dout, aff_cache['aff'+str(i)])
        grads['aff_W'+str(i)] += self.reg * self.params['aff_W'+str(i)]
        
    # reshape dc
    dout = dout.reshape(original_dim)
    
    for i in range(self.conv_num, 0, -1):
    
        #dout = max_pool_backward_naive(dout, pool_cache['conv'+str(i)])
        dout = max_pool_backward_fast(dout, pool_cache['conv'+str(i)])
        dout = relu_backward(dout, reluconv_cache['conv'+str(i)])
        dout, grads['conv_W'+str(i)], grads['conv_b'+str(i)] = conv_backward_fast(dout, conv_cache['conv'+str(i)])
        #dout, dw1, db1 = conv_backward_naive(dout, conv_cache['conv'+str(i)])
        grads['conv_W'+str(i)] += self.reg * self.params['conv_W'+str(i)]
       
    ############################################################################
       
    return loss, grads
  
  
pass
