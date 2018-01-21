import math
import numpy as np

def mini_batch_generator(X, Y, mini_batch_size = 64):
    """
    Creates an generator of random minibatches from (X, Y), caller usage: "for xBatch, yBatch in iter_mini_batches(X, Y, 64):"
    
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
        mini_batch = indices[start_index:start_index + mini_batch_size]
        yield X[mini_batch], Y[mini_batch]

def generator_to_mini_batches(items, mini_batch_size):
    """
    Generate a list of mini batches of size mini_batch_size.
    Some time it is desirable to convert from a generator to slited list.
    Arguments:
    items -- a generator 
    mini_batch_size -- size of each mini bacth
    Output: A list of lists, each with a size of mini_batch_size, except maybe the the mini batch.
    """
    mini_batches = []
    mini_batch = []
    batch_count = 0
    for item in items:
        batch_count += 1
        if batch_count == batch_size:
            mini_batch.append(item)
            mini_batches.append(miniBatch)
            mini_batch = []
            batch_count = 0
        else:            
            mini_batch.append(item)
    # last mini batch has a odd size (leaa than mini_batch_size), but still need to appeded to the batches
    if batch_count != 0:
        mini_batches.append(mini_batch)
    
    return mini_batches

def iter_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (number of examples, Height, Width, Channels), number of example is X.shape[0] 
    Y -- ground truth "labels" of shape (number of examples, label). number of example is Y.shape[0]  
    mini_batch_size - size of the desired mini-batches, integer
    seed -- In case you need a repeatable mini_batch for debugging, user can pass random seed in if desired.
    
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


def convert_to_one_hot(Y, number_classes):
    """
    Creates a one hot version of matrix Y 
    
    Arguments:
    Y -- typically just a one dimension "label" vector (containing just the actual class number)
    number_classes -- numberOfClasses determines the size of I matrix to create for each label.
    
    Returns:
    Y -- the one hot version matrix of original Y vector of shape [numberOfLabels, numberOfClasses]
    """
    Y = np.eye(number_classes)[Y.reshape(-1)]
    return Y

