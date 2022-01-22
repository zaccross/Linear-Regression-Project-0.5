"""
Zac Cross
Exp02 for linear regression project
This file contains the classifer for the MNIST data
set with Linear Regression Model
"""
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np


class Exp02:
    """ This class hold the experiment to be ran """

    @staticmethod
    def load_train_test_data(file_path_prefix=""):
        """
        This method loads the training and testing data
        :param file_path_prefix: Any prefix needed to correctly locate the files.
        :return: x_train, y_train, x_test, y_test, which are to be numpy arrays.
        x_train and x_test should have a .shape value of (#samples, 28, 28)
        y_train and y_test should have a shape of (#samples, ) or (#samples, 1)
        """
        # Following the format from exp00:
        # Read in the files to np-array where each line is a list of comma separated vals
        load_train = np.loadtxt(file_path_prefix + "mnist_train.csv", delimiter=',')
        load_test = np.loadtxt(file_path_prefix + "mnist_test.csv", delimiter=',')

        # Reshape the data into (label, 28x28 matrix of pixels)
        # We'll make a function since we need to reshape multiple datasets
        reshaped_train = [(line[0], line[1:]) for line in load_train]
        reshaped_test = [(line[0], line[1:]) for line in load_test]

        # Now we can separate the pairs into the labels (Y) and matrix data (X)
        x_train = [pair[1] for pair in reshaped_train]
        y_train = [pair[0] for pair in reshaped_train]
        x_test = [pair[1] for pair in reshaped_test]
        y_test = [pair[0] for pair in reshaped_test]
        print(y_train[1])
        print(x_train[1])
        return x_train, y_train, x_test, y_test

    @staticmethod
    def compute_mean_error_rate(true_y_values, predicted_y_values):
        """
        Computes the
        :param true_y_values:
        :param predicted_y_values:
        :return: The mean error rate of true values vs predicted values.
        """
        # Complete me!
        total = 0
        length = len(true_y_values)
        for i in range(length):
            if true_y_values[i] != int(round(predicted_y_values[i])):
                total += 1

        return total / length

    @staticmethod
    def print_error_report(trained_model, x_train, y_train, x_test, y_test):
        """ Prints error report """
        print("\tEvaluating on Training Data")
        # Evaluating on training data is a less effective as an indicator of
        # accuracy in the wild. Since the model has already seen this data
        # before, it is a lessrealistic measure of error when given novel/unseen
        # inputs.
        #
        # The utility is in its use as a "sanity check" since a trained model
        # which preforms poorly on data it has seen before/used to train
        # indicates underlying problems (either more data or data preprocessing
        # is needed, or there may be a weakness in the model itself.

        y_train_pred = trained_model.predict(x_train)

        mean_error_rate_train = Exp02.compute_mean_error_rate(y_train, y_train_pred)

        print("\tMean Error Rate (Training Data):", mean_error_rate_train)
        print()

        print("\tEvaluating on Testing Data")
        # Is a more effective as an indicator of accuracy in the wild.
        # Since the model has not seen this data before, so is a more
        # realistic measure of error when given novel inputs.

        y_test_pred = trained_model.predict(x_test)

        mean_error_rate_test = Exp02.compute_mean_error_rate(y_test, y_test_pred)

        print("\tMean Error Rate (Testing Data):", mean_error_rate_test)
        print()

    def run(self):
        """ Builds and trains the model"""
        start_time = datetime.now()
        print("Running Exp: ", self.__class__, "at", start_time)

        print("Loading Data")
        x_train, y_train, x_test, y_test = Exp02.load_train_test_data()

        print("Training Model...")

        #######################################################################
        # Complete this 2-step block of code using the variable name 'model' for
        # the linear regression model.
        # You can complete this by turning the given psuedocode to real code
        #######################################################################

        # (1) Initialize model; model = NameOfLinearRegressionClassInScikitLearn()
        model = LinearRegression()  # Fix this line

        # (2) Train model using the function 'fit' and the variables 'x_train'
        # and 'y_train'

        model.fit(x_train, y_train)  # Fix this line

        print("Training complete!")
        print()

        print("Evaluating Model")
        Exp02.print_error_report(model, x_train, y_train, x_test, y_test)

        # End and report time.
        end_time = datetime.now()
        print("Exp is over; completed at", datetime.now())
        total_time = end_time - start_time
        print("Total time to run:", total_time)
