import numpy as np


def construct_matrix(first_array, second_array):
    return np.hstack((first_array.reshape(len(first_array), 1), second_array.reshape(len(second_array), 1)))
