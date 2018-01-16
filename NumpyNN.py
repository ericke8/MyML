import numpy as np

"""
This design adopt from Coursera Introduction to Deep Learning course
"""

class Layer:
    """
    A building block. Base class for other concrete layers. Each layer is capable of performing two things:
    
    - Process input to get output:           output = layer.forward(input)
    
    - Propagate gradients through itself:    grad_input = layer.backward(input, grad_output)
    
    Some layers also have learnable parameters which they update during layer.backward.
    """
    def __init__ (self):
        """Here you can initialize layer parameters (if any) and auxiliary stuff."""
        # A dummy layer does nothing
        pass
    
    def forward(self, input):
        """
        Takes input data of shape [batch, numFeatures], returns output data [batch, numClasses]
        """
        # A dummy layer just returns whatever it gets as input.
        return input

    def backward(self,input, grad_output):
        """
        Performs a backpropagation step through the layer, with respect to the given input.
        
        To compute loss gradients w.r.t input, you need to apply chain rule (backprop):
        
        d loss / d x  = (d loss / d layer) * (d layer / d x)
        
        Luckily, you already receive d loss / d layer as input, so you only need to multiply it by d layer / d x.
        
        If your layer has parameters (e.g. dense layer), you also need to update them here using d loss / d layer
        """
        # The gradient of a dummy layer is precisely grad_output, but we'll write it more explicitly
        num_units = input.shape[1]
        
        d_layer_d_input = np.eye(num_units)
        
        return np.dot(grad_output, d_layer_d_input) # chain rule

class ReLU(Layer):
    def __init__(self):
        """ReLU layer simply applies elementwise rectified linear unit to all inputs"""
        pass
    
    def forward(self, input):
        """Apply elementwise ReLU to [batch, numFeatures] matrix"""
        return np.maximum(input, 0)
    
    def backward(self, input, grad_output):
        """Compute gradient of loss w.r.t. ReLU input"""
        relu_grad = input > 0
        return grad_output*relu_grad        

class Dense(Layer):
    """
    regular dense fully connect network. linear part (without activation)
    """
    def __init__(self, numFeatures, output_units, learning_rate=0.1, init='Default'):
        """
        A dense layer is a layer which performs a learned affine transformation:
        f(x) = <W*x> + b
        """
        self.learning_rate = learning_rate
        
        # initialize weights with small random numbers. We use normal initialization, 
        # Add the Xavier factor for init. could add more later.
        if init == 'Xavier':
            self.weights = np.random.randn(numFeatures, numClasses) * np.sqrt(2.0 / (numFeatures + numClasses))
        else:
            self.weights = np.random.randn(numFeatures, numClasses) * 0.05
        self.biases = np.zeros(output_units)
        
    def forward(self,input):
        """
        Perform an affine transformation:
        f(x) = <W*x> + b, where W is of shape[numFeatures, numClasses], and b [batchSize, 1] to be broadcasted
        input shape: [batchSize, numFeatures]
        output shape: [batchSize, numClasses]
        """
        return np.dot(input, self.weights) + self.biases 
    
    def backward(self,input,grad_output):
        """
        Perform back propogation
        dw = 
        """
        # Since grad_input (d f / d x) = grad_output (d f / d dense)  * grad_layer_input (d dense / d x)
        # where d dense/ d x = weights transposed as dense = weight * x + bias
        # Keep in minde grad_output shape must be the same as that of output, so does grad_input shape        
        grad_input =  np.dot(grad_output, self.weights.T)
        
        # compute gradient w.r.t. weights and biases
        # Since grad_weight (d f / d w) = grad_output (d f / d dense)  * grad_layer_weight (d dense / d w)
        # where d dense/ d w = inputs transposed as dense = weight * x + bias
        # similarly, d dense / d b = 1, just need to average for all classes, use axis=0 to averge the classes. 
        grad_weights = 1 / (input.shape[0]) * np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis = 0)
        
        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        # Here we perform a stochastic gradient descent step. 
        # Later on, you can try replacing that with something better.
        self.weights = self.weights - self.learning_rate*grad_weights
        self.biases = self.biases - self.learning_rate*grad_biases
        
        return grad_input


class LogicTrainer
    def softmax_crossentropy_with_logits(logits,reference_answers):
        # Compute crossentropy from logits[batch, numClasses] and ids of correct answers
        logits_for_answers = logits[np.arange(len(logits)),reference_answers]
        xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits),axis=-1))
        
        return xentropy
    
    def grad_softmax_crossentropy_with_logits(logits,reference_answers):
        # Compute crossentropy gradient from logits[batch, n_classes] and ids of correct answers
        ones_for_answers = np.zeros_like(logits)
        ones_for_answers[np.arange(len(logits)),reference_answers] = 1
        
        softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
        
        return - ones_for_answers + softmax   
    def forward(network, X):
        """
        Compute activations of all network layers by applying them sequentially.
        Return a list of activations for each layer. 
        Make sure last activation corresponds to network logits.
        """
        activations = []
        input = X
    
        for eachLayer in network:
            activations.append(eachLayer.forward(input)) # push the output to activations cache 1st
            input = activations[-1]                      # assign last item to input, ready for next layer
            
        assert len(activations) == len(network)
        return activations
    
    def predict(network,X):
        """
        Compute network predictions.
        """
        logits = forward(network,X)[-1]
        return logits.argmax(axis=-1)
    
    def train(network,X,y):
        """
        Train network on a given batch of X and y.
        First need to call forward function of this class to get all layer activations.
        Then run backward of each layer going from last to first layer.
        
        After you called backward for all layers, all Dense layers have already made one gradient step.
        """
        
        # Get the layer activations
        layer_activations = forward(network,X)
        layer_inputs = [X]+layer_activations  #layer_input[i] is an input for network[i]
        logits = layer_activations[-1]
        
        # Compute the loss and the output gradient
        loss = softmax_crossentropy_with_logits(logits,y)
        loss_grad = grad_softmax_crossentropy_with_logits(logits,y)
        #print("loss has shape" + str(loss.shape))
        
        
        # Back propagate gradients through the network
        gradCurrentLayer = loss_grad
        for l in range(len(network))[::-1]:
            # My Comment: Layer l input is the layer l-1 Output, layer l backward produce its grad input, which is l-1 gradOutput
            gradCurrentLayer = network[l].backward(layer_inputs[l], gradCurrentLayer)
            
        return np.mean(loss)    
