import numpy as np

class LaplaceDistribution:
    @staticmethod
    def mean_abs_deviation_from_median(x: np.ndarray):
        '''
        Args:
        - x: A numpy array of shape (n_objects, n_features) containing the data
          consisting of num_train samples each of dimension D.
        '''
        # Calculate the median for each feature (column)
        medians = np.median(x, axis=0)

        # Calculate the absolute deviations from the median for each feature
        abs_deviations = np.abs(x - medians)

        # Calculate the mean of the absolute deviations for each feature
        mean_abs_deviation = np.mean(abs_deviations, axis=0)

        return mean_abs_deviation

    def __init__(self, features):
        '''
        Args:
            feature: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        # loc -> location of the distribution
        # scale -> scale of the distribution
        # You need to calculate the mean_abs_deviation_from_median and use it to initialize self.loc and self.scale
        self.loc = np.median(features, axis=0)  # Median of each feature
        self.scale = self.mean_abs_deviation_from_median(features)  # Mean absolute deviation from median for each feature
        ####

    def logpdf(self, values):
        '''
        Returns logarithm of probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        return -np.log(2 * self.scale) - np.abs(values - self.loc) / self.scale
        ####
        
    def pdf(self, values):
        '''
        Returns probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        return np.exp(self.logpdf(values))