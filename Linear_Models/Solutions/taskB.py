import numpy as np

def root_mean_squared_logarithmic_error(y_true, y_pred, a_min=1.0) -> float:
    assert len(y_true) == len(y_pred), "The length of true values and predicted values must be the same."
    assert np.all(y_true >= 0), "All true values must be non-negative."
    # replace all values less than a_min with a_min
    y_pred = np.maximum(y_pred, a_min)  # replace all values less than 1 with 1
    
    # Calculate the logarithmic differences
    log_diff = np.log(y_pred) - np.log(y_true)
    
    # Calculate the mean squared logarithmic error
    msle = np.mean(log_diff ** 2)
    
    # Return the root of the mean squared logarithmic error
    return np.sqrt(msle)
