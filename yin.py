import numpy as np
import librosa 



def _diff(x1,x2):
    return np.sum((x1-x2)**2)
def _autocorr(x1,x2):
    return np.sum(x1*x2)

def yin(x,step=4,threshold=0.1):
    """
    x: frame to be treated

    returns: Autocorrelation function of x
                (at t=0)
    
    """
    if step == 1:
        func = _autocorr
    else:
        func = _diff

    W = x.shape[0]//2
    min_lag = 0
    lags = range(min_lag,W)
    
    result = np.zeros(x.shape[0]//2)
    if step >= 3:
        acc_norm = 1e-8
    for lag in lags:
        x1 = x[0:W]
        x2 = x[lag:lag+W]
        result[lag] = func(x1,x2)
        
        if step >= 3:
            acc_norm += result[lag]
            if lag == 0:
                result[lag] = 1
            else:
                norm = acc_norm/lag
                result[lag] /= norm
    
    if step == 1:
        f0 = np.argmax(result)
        return f0,result
    elif step <= 3:
        f0 = np.argmin(result[5:]) # 5 is just a little threshold 
        return f0+5,result         # not to get the first peak.
    elif step == 4:
        for i,r in enumerate(result):
            if result[i] <= result[i-1] \
                and result[i] <= result[i+1]:
                if r <= threshold:
                    f0 = i
                    return f0,result
                
        return 0,result
    else: 
        print("Step not yet implemented.")
        return 0,None
            
def yin_signal(x,hop=2048,step=4):
    """
    yin algorithm for a whole signal.
    Painfully inneficient.
    Could be made much more efficient using 
    caching from previous frames.
    
    """
    f0s = []
    ind = 0
    frame = x[ind:ind+2048]
    while frame.shape[0] == 2048:
        f0,_ = yin(frame,step=step)
        f0s.append(f0)
        ind += hop
        frame = x[ind:ind+2048]
    return np.array(f0s)