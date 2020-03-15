import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W

    return loss, grad

def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    probs = predictions.copy()
    if len(predictions.shape) == 1:
        probs -= np.max(predictions)
        probs = np.exp(probs) / np.sum(np.exp(probs))
    else:
        probs -= np.max(predictions, axis=1).reshape(-1, 1)
        probs = np.exp(probs) / np.sum(np.exp(probs), axis=1).reshape(-1, 1)

    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    if type(target_index) == int:
        return -np.log(probs[target_index])
    else:
        return - np.mean(np.log(probs[range(target_index.shape[0]), target_index]))


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    soft = softmax(predictions)
    loss = cross_entropy_loss(soft, target_index)
    dprediction = soft
    if type(target_index) == int:
        dprediction[target_index] -= 1
        return loss, dprediction
    else:
        dprediction[range(target_index.shape[0]), target_index] -= 1
        return loss, dprediction / target_index.shape[0]


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, d_out):
        d_result = d_out * (self.X >= 0)
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.sum(d_out, axis=0)[:, np.newaxis].T
        d_input = np.dot(d_out, self.W.value.T)
        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        self.X = np.pad(X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        batch_size, height, width, channels = X.shape

        out_height = height - self.filter_size + 2 * self.padding + 1
        out_width = width - self.filter_size + 2 * self.padding + 1

        result = np.zeros((batch_size, out_height, out_width, self.out_channels))
        for y in range(out_height):
            for x in range(out_width):
                tempI = (self.X[:, y:(y + self.filter_size), x:(x + self.filter_size), :, np.newaxis] *
                        self.W.value[np.newaxis, :])
                tempI = tempI.sum(axis=(3, 2, 1))
                result[:, y, x] = tempI + self.B.value

        return result

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        dx = np.zeros_like(self.X)
        for y in range(out_height):
            for x in range(out_width):
                tempI = self.X[:, y:(y + self.filter_size), x:(x + self.filter_size), :, np.newaxis]
                d_out_g = d_out[:, y, x, np.newaxis, np.newaxis, np.newaxis, :]
                self.W.grad += np.sum(d_out_g * tempI, axis=0)
                dx[:, y:(y + self.filter_size), x:(x + self.filter_size), :] += np.sum(self.W.value * d_out_g, axis=4)

        self.B.grad += np.sum(d_out, axis=(0, 1, 2))

        height -= 2 * self.padding
        width -= 2 * self.padding
        d_input = dx[:, self.padding:(self.padding + height), self.padding:(self.padding + width), :]
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X.copy()
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        result = np.zeros((batch_size, out_height, out_width, channels))
        for y in range(out_height):
            for x in range(out_width):
                tempI = self.X[:, y:(y + self.pool_size), x:(x + self.pool_size), :]
                result[:, y, x] = tempI.max(axis=(2, 1))

        return result

    def backward(self, d_out):
        _, out_height, out_width, _ = d_out.shape
        result = np.zeros_like(self.X)
        for y in range(out_height):
            for x in range(out_width):
                tempI = self.X[:, y:(y + self.pool_size), x:(x + self.pool_size), :]
                tempI = (tempI == tempI.max(axis=(2, 1))[:, np.newaxis, np.newaxis, :])
                dx = (d_out[:, y, x, :])[:, np.newaxis, np.newaxis, :]
                result[:, y:(y + self.pool_size), x:(x + self.pool_size), :] += dx * tempI
        return result

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        self.X_shape = X.shape
        batch_size, height, width, channels = X.shape
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        batch_size, height, width, channels = self.X_shape
        return d_out.reshape(batch_size, height, width, channels)

    def params(self):
        # No params!
        return {}
