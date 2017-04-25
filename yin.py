import numpy as np

def autocorrelation_function(x,t,lag):
    """
    x: signal to be treated

    returns: Difference function of x
    
    """
    W = 600

    max_lag = x.shape[0]//2
    min_lag = 0

    x_pad = np.hstack(x,np.zeros((x.shape)))

    r = 0

    x1 = x[t+1:t+W]
    x2 = x[t+1+lag:t+W+lag]

    return np.sum(x1*x2)


