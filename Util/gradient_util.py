import numpy as np

def eval_numerical_gradient(function, x, verbose=False, h=0.00001):
    """Evaluates gradient df/dx via finite differences:
    df/dx ~ (f(x+h) - f(x-h)) / 2h
    Adopted from coursera advanced machine learning course.
    
    Arguments:
    function -- function definition, typical passed in using lambda expression. for instance: "f = lambda x: x+2" is f(x)=x+2 
    x        -- variable matrix interested for gradient checking
    h        -- step size 
    """
    assert h > 0.0
    #f_x = function(x) # will evaluate function value at original point x
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # evaluate function at x+h
        ix = it.multi_index
        oldValue = x[ix]
        x[ix] = oldValue + h # increment by h
        f_xPlusH = function(x) # evalute f(x + h)
        x[ix] = oldValue - h
        f_xMinusH = function(x) # evaluate f(x - h)
        x[ix] = oldValue # restore, as we donot want to alter x

        # compute the partial derivative with centered formula
        grad[ix] = (f_xPlusH - f_xMinusH) / (2 * h) # the slope
        if verbose:
            print (ix, grad[ix])
        it.iternext() # step to next dimension

    return grad
