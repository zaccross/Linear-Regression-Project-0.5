"""
Exp 00 - Tests Data preprocessing and trains a basic linear regression
            Model for abalone age prediction
"""

from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np


class Exp00:

    """ Experiment Class to test and run abalone data processing
        And linear model training """

    @staticmethod
    def load_train_test_data(file_path_prefix=""):
        """
        This method loads the training and testing data
        :param file_path_prefix: Any prefix needed to correctly locate the files.
        :return: x_train, y_train, x_test, y_test, which are to be numpy arrays.
        """

        train = np.loadtxt(file_path_prefix + "abalone_train.csv", delimiter=',')
        test = np.loadtxt(file_path_prefix + "abalone_test.csv", delimiter=',')

        x_train = train[:,:-1]
        y_train = train[:, -1]

        x_test = test[:, :-1]
        y_test = test[:, -1]

        return x_train, y_train, x_test, y_test

    @staticmethod
    def compute_mean_absolute_error(true_y_values, predicted_y_values):
        """Computes mean error """
        list_of_errors = []
        for true_y, pred_y in zip(true_y_values, predicted_y_values):
            error = abs(true_y - pred_y)
            list_of_errors.append(error)
        mean_abs_error = np.mean(list_of_errors)
        return mean_abs_error

    @staticmethod
    def compute_mean_absolute_percentage_error(true_y_values, predicted_y_values):
        """ Computes the mean absolute percentage error """
        list_of_perc_errors = []
        for true_y, pred_y in zip(true_y_values, predicted_y_values):
            error = abs((true_y - pred_y) / true_y)
            list_of_perc_errors.append(error)
            list_of_perc_errors.append(error)
        mean_abs_error = np.mean(list_of_perc_errors)
        return mean_abs_error

    @staticmethod
    def print_error_report(trained_model, x_train, y_train, x_test, y_test):
        """ Prints the error report """
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

        mean_absolute_error_train = Exp00.compute_mean_absolute_error(y_train, y_train_pred)
        mean_absolute_perc_error_train = Exp00.compute_mean_absolute_percentage_error(y_train,
                                                                                      y_train_pred)

        print("\tMean Absolute Error (Training Data):", mean_absolute_error_train)
        print("\tMean Absolute Percentage Error (Training Data):", mean_absolute_perc_error_train)
        print()

        print("\tEvaluating on Testing Data")
        # Is a more effective as an indicator of accuracy in the wild.
        # Since the model has not seen this data before, so is a more
        # realistic measure of error when given novel inputs.

        y_test_pred = trained_model.predict(x_test)

        mean_absolute_error_test = Exp00.compute_mean_absolute_error(y_test, y_test_pred)
        mean_absolute_perc_error_test = Exp00.compute_mean_absolute_percentage_error(y_test,
                                                                                     y_test_pred)

        print("\tMean Absolute Error (Testing Data):", mean_absolute_error_test)
        print("\tMean Absolute Percentage Error (Testing Data):", mean_absolute_perc_error_test)
        print()

    def run(self):
        """ Runs the training """
        start_time = datetime.now()
        print("Running Exp: ", self.__class__, "at", start_time)

        print("Loading Data")
        x_train, y_train, x_test, y_test = Exp00.load_train_test_data()

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
        Exp00.print_error_report(model, x_train, y_train, x_test, y_test)

        # End and report time.
        end_time = datetime.now()
        print("Exp is over; completed at", datetime.now())
        total_time = end_time - start_time
        print("Total time to run:", total_time)
