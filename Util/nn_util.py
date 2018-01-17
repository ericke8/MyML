import math
import numpy as np

def iter_mini_batches(X, Y, mini_batch_size = 64):
    """
    Creates an iterable of random minibatches from (X, Y), caller usage: "for xyBatch in iter_mini_batches(X, Y, 64):"
    
    Arguments:
    X -- input data, of shape (number of examples, Height, Width, Channel, ...), as long as number of example is X.shape[0] 
    Y -- ground truth "labels" of shape (number of examples, label.shape ). number of example in Y.shape[0]  
    mini_batch_size - size of the desired mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    assert X.shape[0] == Y.shape[0]
    m = X.shape[0]                  # number of training examples
    
    #shuffle the indice
    indices = np.arange(inputs.shape[0])
    np.random.shuffle(indices)
    
    # use yield keyword to create the generator for caller's iteration
    for start_index in range(0, m - mini_batch_size + 1, mini_batch_size):
        miniBatch = indices[start_index:start_index + mini_batch_size]
        yield X[miniBatch], Y[miniBatch]

def iter_mini_batches(X, Y, mini_batch_size = 64, , seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (number of examples, Height, Width, Channels), number of example is X.shape[0] 
    Y -- ground truth "labels" of shape (number of examples, label). number of example is Y.shape[0]  
    mini_batch_size - size of the desired mini-batches, integer
    seed -- In case you need a repeatable random result, user can pass random number in if true random is desired.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    assert X.shape[0] == Y.shape[0]
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def convert_to_one_hot(Y, numberOfClasses):
    """
    Creates a one hot version of matrix Y 
    
    Arguments:
    Y -- typically just a one dimension "label" vector (containing just the actual class number)
    numberOfClasses -- numberOfClasses determines the size of I matrix to create for each label.
    
    Returns:
    Y -- the one hot version matrix of original Y vector
    """
    Y = np.eye(C)[numberOfClasses.reshape(-1)].T
    return Y

