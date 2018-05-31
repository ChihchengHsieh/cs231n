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
        - filter_size: Width/height of filters to use in the convolutional layer
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
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #                           
        ############################################################################
        depth, H,W = input_dim
        std = weight_scale
        
        
        S = 1 # Stride
        p = (filter_size -1)/ 2 #if the stride = 1, this pad can help the conv to hold the size, input_size = output_size
        self.params['W1'] = std*np.random.randn(num_filters,depth,filter_size,filter_size)
        
        #the output size from the first Conv layer, the relu will give the same size
        Hh = int(1 + (H + 2 * p - filter_size) / S)
        Hw = int(1 + (W + 2 * p - filter_size) / S)
        #size after the first Conv = (N,num_filters,Hh,Hw)
        
        Wp = 2 # width of pooling 
        Hp = 2 # hight of pooling
        Sp = 2 # stride of pooling
        
        #the size after the max pooling
        Hl = int((Hh - Hp) / Sp + 1)
        Wl = int((Hw - Wp) / Sp + 1)
        #the size will be (N,num_filters, Hl,Wl)
        
        
        # So the size for the first affline will be (N, num_filters*Hl*Wl)
        self.params['W2'] = std*np.random.randn(num_filters*Hl*Wl,hidden_dim)
        #the size after the first affline = (N,hidden_dims)]
        
        self.params['W3'] = std*np.random.randn(hidden_dim,num_classes)
        #size after second affline = (N,numclasses)
        
        self.params['b1'] = np.zeros(num_filters)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['b3'] = np.zeros(num_classes)
        
        
        
        
        pass
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

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        h, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        N,F,Hh,Hw = h.shape
        #h = h.reshape(N,F*Hh*Hw)
        h, affine_first_cache = affine_relu_forward(h,W2,b2)
        scores, affine_second_cache =affine_forward(h, W3, b3)
        
        
        
        
        pass
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
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        N,C,H,W = X.shape
        
        data_loss, dout = softmax_loss(scores, y)
        reg_loss = 0.5* self.reg*np.sum(W1**2)
        reg_loss += 0.5* self.reg*np.sum(W2**2)
        reg_loss += 0.5* self.reg*np.sum(W3**3)
        loss = data_loss+ reg_loss
        
        dout, dW3, db3 = affine_backward(dout, affine_second_cache)
        dout, dW2, db2 = affine_relu_backward(dout, affine_first_cache)
        dout.reshape(N,F,Hh,Hw)
        dx, dW1, db1 = conv_relu_pool_backward(dout, conv_cache)
        
        dW3 += self.reg*W3
        dW2 += self.reg*W2
        dW1 += self.reg*W1
        
        grads['W1']= dW1
        grads['W2']= dW2
        grads['W3']= dW3
        grads['b1']= db1
        grads['b2']= db2
        grads['b3']= db3
        
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
