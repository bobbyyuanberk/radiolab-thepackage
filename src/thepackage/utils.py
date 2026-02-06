def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def rms(x):
    """
    Compute root-mean-square of a 1D array.

    Parameters
    ----------
    x : array-like
        Input signal

    Returns
    -------
    float
        RMS value
    """
    return (sum(v*v for v in x) / len(x)) ** 0.5