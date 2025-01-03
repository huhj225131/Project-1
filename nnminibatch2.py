import numpy as np
import h5py
import matplotlib.pyplot as plt

def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layer_dims[l] , layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.random.randn(layer_dims[l], 1)
        ### END CODE HERE ###

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))


    return parameters

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """

    A = 1/(1+np.exp(-Z))
    cache = Z

    return A, cache

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A =  np.maximum(0,Z)

    assert(A.shape == Z.shape)

    cache = Z
    return A, cache

def softmax(Z):
    """
    Implements the softmax activation function in numpy.

    Arguments:
    Z -- numpy array of any shape (e.g., (n_classes, n_samples) for a batch of predictions)

    Returns:
    A -- output of softmax(Z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    # Shift Z for numerical stability (prevent overflow)
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    assert A.shape == Z.shape, "Output shape must match input shape"

    cache = (Z, A)
    return A, cache

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    ### START CODE HERE ### (≈ 1 line of code)
    Z = np.dot(W, A) +b
    ### END CODE HERE ###

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        ### START CODE HERE ### (≈ 2 lines of code)
        Z , linear_cache= linear_forward(A_prev, W, b)
        A , activation_cache = sigmoid(Z)
        ### END CODE HERE ###

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        ### START CODE HERE ### (≈ 2 lines of code)
        Z , linear_cache= linear_forward(A_prev, W, b)
        A , activation_cache = relu(Z)
        ### END CODE HERE ###
    elif activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)
    elif activation == "linear":
        Z , linear_cache = linear_forward(A_prev, W, b)
        A , activation_cache = Z,Z

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache
def L_model_forward(X, parameters, activations):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    # for l in range(1, L):
    #     A_prev = A
    #     ### START CODE HERE ### (≈ 2 lines of code)
    #     A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b' + str(l)], 'relu')
    #     caches.append(cache)
    #     ### END CODE HERE ###

    # # Implement LINEAR -> SOFTMAX. Add "cache" to the "caches" list.
    # ### START CODE HERE ### (≈ 2 lines of code)
    # AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b' + str(L)], 'softmax')
    # caches.append(cache)
    # ### END CODE HERE ###
    for l in range(1, L+ 1):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' +str(l)], activations[l])
        caches.append(cache)
    AL = A
    assert(AL.shape[1] == X.shape[1])

    return AL, caches

def compute_cost(AL, Y, last_activation):
    """
    Compute the cross-entropy cost for softmax output.

    Arguments:
    AL -- probability matrix from softmax, shape (num_classes, number of examples)
    Y -- true labels, shape (1, number of examples), with values from 0 to num_classes-1

    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]  # number of examples

    # Convert Y to one-hot encoding
    if last_activation == 'softmax':
        Y_one_hot = np.eye(AL.shape[0])[:, Y.flatten()]  # Shape: (num_classes, number of examples)
    
        # Compute cross-entropy cost
        cost = -1 / m * np.sum(Y_one_hot * np.log(AL + 1e-8))  # Add small value to avoid log(0)
    elif last_activation == 'linear':
        cost = 1 / m * np.sum((AL - Y)**2 + 1e-8)
    elif last_activation == 'sigmoid':
        cost = -1 / m * np.sum(Y * np.log(AL + 1e-8) + (1 - Y) * np.log(1 - AL + 1e-8))
    elif last_activation == 'relu':
        cost = 1 / m * np.sum((AL - Y)**2 + 1e-8)
    cost = np.squeeze(cost)  # Ensure cost is a scalar
    assert(cost.shape == ())

    return cost

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    ### START CODE HERE ### (≈ 3 lines of code)
    dW = np.dot(dZ, A_prev.T)/ m
    db = np.sum(dZ, axis = 1, keepdims = True) / m    
    dA_prev = np.dot(W.T, dZ)
    ### END CODE HERE ###
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ
def softmax_backward(dA, cache):
    """
    Implement the backward propagation for a single SOFTMAX unit.

    Arguments:
    dA -- Gradient of the cost with respect to the output of the softmax, shape (n_classes, m)
    cache -- 'Z' where we store for computing backward propagation efficiently, shape (n_classes, m)

    Returns:
    dZ -- Gradient of the cost with respect to Z, shape (n_classes, m)
    """
    Z, A = cache  # Retrieve Z, A from the cache
    
    # Compute the softmax activation (forward computation)
    
    
    # Compute the gradient using the Jacobian
    dZ = np.empty_like(Z)  # Placeholder for dZ
    for i in range(Z.shape[1]):  # Loop through each sample
        s = A[:, i].reshape(-1, 1)  # Softmax probabilities for the i-th sample
        jacobian = np.diagflat(s) - np.dot(s, s.T)  # Compute Jacobian matrix
        dZ[:, i] = np.dot(jacobian, dA[:, i])  # Multiply Jacobian by dA for this sample

    # Ensure the shape of dZ is the same as Z
    assert (dZ.shape == Z.shape)
    
    return dZ
# def linear_activation_backward(dA, cache):
#     Z = cache
#     dZ = dA  
#     assert (dZ.shape == Z.shape)
    
#     return dZ
def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = relu_backward(dA, activation_cache) 
        dA_prev, dW, db=linear_backward(dZ, linear_cache)
        ### END CODE HERE ###
        
    elif activation == "sigmoid":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = sigmoid_backward(dA, activation_cache) 
        dA_prev, dW, db=linear_backward(dZ, linear_cache)
        ### END CODE HERE ###
    elif activation == "softmax":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = softmax_backward(dA, activation_cache) 
        dA_prev, dW, db=linear_backward(dZ, linear_cache)
        ### END CODE HERE ###
    elif activation == "linear":
        dZ = dA 
        dA_prev, dW, db=linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches, activations):
    grads = {}
    L = len(caches)
    m  = AL.shape[1]
    if activations[-1] == 'softmax':
        Y_one_hot = np.eye(AL.shape[0])[:, Y.flatten()]  # Shape: (num_classes, number of examples)
        dAL = AL - Y_one_hot
    elif activations[-1] in ['relu', 'linear', 'sigmoid']:
        dAL = AL - Y
    grads['dA' +str(L)] = dAL
    for l in reversed(range(L)):
        grads['dA' + str(l )] , grads['dW' + str(l+ 1)], grads['db' +str(l+ 1)] = linear_activation_backward(grads['dA' +str(l + 1)], caches[l ], activations[l + 1])
        
    return grads
def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    ### START CODE HERE ### (≈ 3 lines of code)
    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate*grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" +str(l)] - learning_rate *grads["db" + str(l)]
    
    ### END CODE HERE ###
        
    return parameters

def L_layer_model(X, Y, parameters,activations, learning_rate = 0.0075):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    
    
  

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SOFTMAX.
        ### START CODE HERE ### (≈ 1 line of code)
    AL, caches = L_model_forward(X, parameters,activations)
        ### END CODE HERE ###
        
        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
    cost = compute_cost(AL, Y, activations[-1])
        ### END CODE HERE ###
    
        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
    grads = L_model_backward(AL, Y, caches, activations)
        ### END CODE HERE ###
 
        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
    parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
                
        # Print the cost every 100 training example
    
            
    # plot the cost
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per tens)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()
    
    return parameters, cost

def random_mini_batches(X, Y, mini_batch_size, seed=0):
    """
    Creates a list of random mini-batches from (X, Y)
    
    Arguments:
    X -- input data, numpy array of shape (number of examples, number of features)
    Y -- true "label" vector of shape (number of examples, )
    mini_batch_size -- size of the mini-batches
    seed -- seed for randomization
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    np.random.seed(seed)
    m = X.shape[0]  # number of training examples
    mini_batches = []
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]  

    # Step 2: Partition (X, Y). Minus the end case.
    num_complete_minibatches = m // mini_batch_size
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : (k + 1) * mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : (k + 1) * mini_batch_size, :]
        mini_batches.append((mini_batch_X, mini_batch_Y))
    
    # Handle the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size :, :]
        mini_batches.append((mini_batch_X, mini_batch_Y))
    
    return mini_batches

def train(X, y,parameters, activations, mini_batch_size=32, epochs = 20, learning_rate = 0.2 ):
    
    total_cost = []
    
    for i in range(epochs):
        mini_batches = random_mini_batches(X, y, mini_batch_size)
        batch_number = len(mini_batches) / 2
        cost_per_epoch = 0
        for mini_batch in mini_batches:
            X_mini , y_mini = mini_batch
            parameters, cost = L_layer_model(X_mini.T, y_mini.T, parameters, activations, learning_rate )
            cost_per_epoch += cost
        total_cost.append(cost_per_epoch / batch_number)
        print(cost_per_epoch / batch_number)
    return parameters, total_cost