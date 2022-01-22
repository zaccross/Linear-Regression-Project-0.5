"""
Zac Cross
CIS 372m
This file contains a graphical test for MNIST data
using matplotlib's imshow()
"""
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


class Exp01:
    """ Experiment 1 Test if my code graphs MNIST Data"""

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

        def _reshape(data):
            """ Inputs an array of lists with a [label, pixel 0, .... , pixel 783]
                Reshapes the data to a list of (label, 28x28 matrix) pairs """
            reshaped_data = []

            # Go through the list, line by line
            # and make 28 pixel columns for the pixel matrix
            # and return the label matrix pair.
            for line in data:
                matrix = []
                for i in range(28):
                    col = line[i * 28 + 1:(i + 1) * 28 + 1]
                    matrix.append(col)
                reshaped_data.append((line[0], matrix))
            return reshaped_data

        reshaped_train = _reshape(load_train)
        reshaped_test = _reshape(load_test)

        # Now we can separate the pairs into the labels (Y) and matrix data (X)
        x_train = [pair[1] for pair in reshaped_train]
        y_train = [pair[0] for pair in reshaped_train]
        x_test = [pair[1] for pair in reshaped_test]
        y_test = [pair[0] for pair in reshaped_test]

        return x_train, y_train, x_test, y_test

    @staticmethod
    def display_image(x_image_data, y_label_value):
        """
        Display an image using matplotlib.
        :param x_image_data: A numpy array of shape (28, 28)
        :param y_label_value: An integer label.
        :return: None; displays a matplotlib plot.
        """
        plt.imshow(x_image_data)
        plt.title(f"Label - {y_label_value}")
        plt.show()

    def run(self):
        """Runs the image grabbing"""
        start_time = datetime.now()
        print("Running Exp: ", self.__class__, "at", start_time)

        print("Loading Data")
        x_train, y_train, x_test, y_test = Exp01.load_train_test_data()

        print("Displaying Certain Figures")

        Exp01.display_image(x_train[0], str(y_train[0]) + " (train)")
        Exp01.display_image(x_train[42], str(y_train[42]) + " (train)")
        Exp01.display_image(x_train[156], str(y_train[156]) + " (train)")

        Exp01.display_image(x_test[0], str(y_test[0]) + " (test)")
        Exp01.display_image(x_test[42], str(y_test[42]) + " (test)")
        Exp01.display_image(x_test[542], str(y_test[542]) + " (test)")

        # End and report time.
        end_time = datetime.now()
        print("Exp is over; completed at", datetime.now())
        total_time = end_time - start_time
        print("Total time to run:", total_time)
