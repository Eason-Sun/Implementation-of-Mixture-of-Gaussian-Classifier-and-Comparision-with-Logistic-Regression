import pandas as pd
import numpy as np
import math
import time


class MixtureOfGaussians:

    # Split the data to two halves by their class labels
    def split_by_class(self, train_data, train_label):
        data_label = pd.concat([train_data, train_label], axis=1).values
        data_label_sorted = data_label[data_label[:, -1].argsort()]
        n_1 = np.searchsorted(data_label_sorted[:, -1], data_label_sorted[0, -1], side='right')
        X_1 = data_label_sorted[:n_1, :-1]
        X_2 = data_label_sorted[n_1:, :-1]
        return X_1, X_2

    # Use MLE to estimate class prior (Pi), class mean (Mu_1 and Mu_2) and covariance matrix (Sigma)
    def fit(self, train_data, train_label):
        X_1, X_2 = self.split_by_class(train_data, train_label)
        # Number of samples for each class
        n_1 = X_1.shape[0]
        n_2 = X_2.shape[0]
        # Calculate the Pi
        self.Pi = n_1 / (n_1 + n_2)
        # Calculate the Mu_1 and Mu_2
        self.Mu_1 = np.mean(X_1, axis=0)
        self.Mu_2 = np.mean(X_2, axis=0)
        # Calculate the Sigma
        X_1_zero_mean = X_1 - np.repeat(np.array([self.Mu_1]), n_1, axis=0)
        X_2_zero_mean = X_2 - np.repeat(np.array([self.Mu_2]), n_2, axis=0)
        S1 = (1 / n_1) * np.matmul((np.transpose(X_1_zero_mean)), X_1_zero_mean)
        S2 = (1 / n_2) * np.matmul(np.transpose(X_2_zero_mean), X_2_zero_mean)
        self.Sigma = self.Pi * S1 + (1 - self.Pi) * S2

    # Return test accuracy for a particular class label
    def MAP(self, test_data, class_label=1):
        count = 0
        for x_i in test_data:
            # Calculate the Prior*Likelihood for class 1
            p_1 = self.Pi * math.exp((-0.5) * np.matmul(np.matmul(x_i - self.Mu_1, np.linalg.inv(self.Sigma)),
                                                        np.transpose(x_i - self.Mu_1)))
            # Calculate the Prior*Likelihood for class 2
            p_2 = (1 - self.Pi) * math.exp((-0.5) * np.matmul(np.matmul(x_i - self.Mu_2, np.linalg.inv(self.Sigma)),
                                                              np.transpose(x_i - self.Mu_2)))
            # Predict the class which yields maximum posterior
            if class_label == 1:
                count += 1 if p_1 / (p_1 + p_2) > 0.5 else 0
            else:
                count += 1 if p_2 / (p_1 + p_2) > 0.5 else 0
        return count / test_data.shape[0] * 100

    # Return the total accuracy for the test data and individual accuracies for each class label
    def accuracy(self, test_data, test_label):
        X_1, X_2 = self.split_by_class(test_data, test_label)
        class_1_accuracy = self.MAP(X_1, 1)
        class_2_accuracy = self.MAP(X_2, 2)
        total_accuracy = (class_1_accuracy * X_1.shape[0] + class_2_accuracy * X_2.shape[0]) / (
                X_1.shape[0] + X_2.shape[0])
        return total_accuracy, class_1_accuracy, class_2_accuracy

# Merge multiple csv file into one data and one label data frames. Optionally, we can exclude certain files
def merge_train_files(num_of_files, skip=None):
    df_train_data = pd.DataFrame()
    df_train_label = pd.DataFrame()
    for k in range(num_of_files):
        if k == skip:
            continue
        data = pd.read_csv('dataset/trainData' + str(k + 1) + '.csv', header=None)
        df_train_data = df_train_data.append(data, ignore_index=True)
        label = pd.read_csv('dataset/trainLabels' + str(k + 1) + '.csv', header=None)
        df_train_label = df_train_label.append(label, ignore_index=True)
    return df_train_data, df_train_label


# Merge 10 training data and label files
df_train_data, df_train_label = merge_train_files(10)
# Read test data and label files
df_test_data = pd.read_csv('dataset/testData.csv', header=None)
df_test_label = pd.read_csv('dataset/testLabels.csv', header=None)

running_times = []
print('Run 100 times: Approximate wait time: 4 sec')
for i in range(100):
    start_time = time.time()
    # Train the MoG classifier with training dataframes
    clf = MixtureOfGaussians()
    clf.fit(df_train_data, df_train_label)
    # Test the accuracy with testing dataframes
    total_accuracy, class_1_accuracy, class_2_accuracy = clf.accuracy(df_test_data, df_test_label)
    running_times.append(time.time() - start_time)
    if i == 99:
        print('Total Accuracy: {:5.2f}%'.format(total_accuracy))
        print('Class 1: \'5\', Accuracy: {:5.2f}%\nClass 0: \'6\', Accuracy: {:5.2f}%'.format(class_1_accuracy,
                                                                                              class_2_accuracy))

print('\nAverage running time: {:6.4f}s'.format(np.array(running_times).mean()))

print('\nParameters:')
print('Pi:\n', clf.Pi)
print('Mu_1:\n', clf.Mu_1)
print('Mu_2:\n', clf.Mu_2)
print('Sigma:\n', clf.Sigma.diagonal())