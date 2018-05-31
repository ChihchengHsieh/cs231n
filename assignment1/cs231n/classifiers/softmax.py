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
  num_dim = X.shape[1]
  num_classes = W.shape[1]  
  num_train = X.shape[0] 
  for i in range(num_train):
        f = X[i,:].dot(W) # get the socore for every class
        correct_class_score = f[y[i]] # see the score in the correct class
        f -= np.max(f) # f becomes [-666, -333, 0]
        p = np.exp(f) / np.sum(np.exp(f)) # safe to do, gives the correct answer
        for d in range(num_dim):
            for k in range(num_classes):
                if k == y[i]:
                    dW[d,k] += X.T[d,i]*(p[k]-1)
                else:
                    dW[d,k] += X.T[d,i]*p[k]
        
        loss += -np.log(p[y[i]])
        
  loss /= num_train
  loss += 0.5 * reg * np.sum(W**2)
  
  dW /= num_train
  dW += reg * W 
  
  pass
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
  num_trains = X.shape[0]
  scores = np.dot(X,W)
  prob = np.exp(scores)/ np.sum(np.exp(scores),axis = 1,keepdims = True)
  loss = np.sum(-np.log(prob[range(num_trains),y]))
  loss /= num_trains
  loss += 0.5 * reg * np.sum(W**2)
  

  #Gradient
  #To get the all the loss for each element
  dscore = prob  # why do we do the gradient like this
  dscore[range(num_trains),y] -= 1
  dw = np.dot(X.T, dscore)
  dW /= num_trains
  dW += reg * W


  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

