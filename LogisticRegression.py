import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

df = pd.read_csv('smoke_detection_iot.csv') # Read the data.


# Aranging data, deleting unwanted columns, split data in X and Y

def arange_data(df, n):  # df = data, n = sample size.

    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the data frame.
    Y = df['Fire Alarm'].to_numpy()[0:n]  # Split the labels.
    # Drop the unwanted columns:
    df.drop(['CNT'], inplace=True, axis=1)
    df.drop(['Unnamed: 0'], inplace=True, axis=1)
    df.drop(['Fire Alarm'], inplace=True, axis=1)

    # df.drop(['UTC'],axis=1,inplace=True)
    # Convert the data to array and apply min-max scaling.
    design = df.to_numpy()
    maximum_values = []
    minimum_values = []
    for i in range(len(df.columns.values.tolist())):
        minimum = np.min(design[:, i])
        maximum = np.max(design[:, i])
        maximum_values.append(maximum)
        minimum_values.append(minimum)
    for i in range(len(df.columns.values.tolist())):
        design[:, i] = (design[:, i] - minimum_values[i]) / (maximum_values[i] - minimum_values[i])

    design_1 = np.c_[np.ones((np.shape(design)[0], 1)), design]  # Add the DC terms.

    X = design_1[0:n].reshape((n, 14))  # Declare the design matrix.

    Beta = np.random.normal(0, 1, size=(14, 1)).reshape((14, 1))  # Initialize the parameters for GD.

    # Beta = np.zeros((14,1))

    return X, Y, Beta  # Return X set, Y set, and parameters.

#Initialize the train and parameters.
n_train = 20000
l = arange_data(df,n_train)
X_train = l[0]
Y_train = l[1]
param = l[2]


# Define the sigmoid function.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Defining the cross entropy loss function.

# Take the dot product of each row^T with the parameter vector, insert it into the sigmoid function, and sum the
# loss function over all rows.

def cross_entropy(weigth, des, y):
    liste = []

    for i in range(len(des)):
        k = des[i].T.dot(weigth)

        loss = -(y[i] * np.log(sigmoid(k)) + (1 - y[i]) * (1 - np.log(sigmoid(k)))) / np.shape(des)[0]

        liste.append(loss)

    sol = sum(np.array(liste))

    return sol


# Define the Gradient Descent (GD) algorithm. Use the fact that d(loss)/d(beta)=X^T(sigmoid(x)-y).
def GD(X, Y, Beta, lr, n_iter):
    loss_list = []
    # Betha = []
    for i in range(n_iter):
        z = np.dot(X.T, (sigmoid(np.dot(X, Beta)) - Y.reshape(len(Y), 1)))
        Beta = Beta - lr * z
        # Beta = Beta - lr * np.dot(X.T, sigmoid(np.dot(X, Beta))-Y)
        # loss = cross_entropy(Beta, X, Y)
        loss_list.append(z)
        # Betha.append(Beta)
    # idx = np.where(abs(np.array(loss_list)) == abs(np.array(loss_list)).min())
    # loss = loss_list[idx[0][0]]

    return Beta, X, z, loss_list  # [Beta,X ,loss, loss_list]

# Train the model by using Gradient Descent algorithm with learning rate = 0.001 and # of iterations = 1000.
sol = GD(X_train, Y_train, param, 0.001, 2000)

optimum_beta=sol[0]


# Predict the labels (y) usşng finding the optimal parameter vector beta.
# Set the threshold to 0.5 and assign class 1 above the threshold and 0 below the threshold.

def score(X, Beta):
    y = []
    s = sigmoid(np.dot(X, Beta))

    for i in range(len(s)):
        if s[i] >= 0.5:
            y.append(1)
        else:
            y.append(0)
    return np.array(y)

predicted_labels_for_train = score(X_train, optimum_beta)


# Error function. False predicitons/sample size
def error(y_t, y):
    n = len(y)
    k = 0
    for i in range(n):
        if y[i] != y_t[i]:
            k = k + 1
    error = k / n

    return error

train_error = error(predicted_labels_for_train,Y_train)

X_test = arange_data(df,1000)[0]
Y_test = arange_data(df,1000)[1]

predicted_labels_for_test = score(X_test, optimum_beta)
test_error = error(predicted_labels_for_test, Y_test)


# Declare precision and recall.

def precision_recall(X_test, Y_test, Beta):
    s = score(X_test, Beta)
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(s)):
        if s[i] == 1 and Y_test[i] == s[i]:
            tp = tp + 1

        elif s[i] == 1 and Y_test[i] != s[i]:
            fp = fp + 1
        elif s[i] == 0 and Y_test[i] != s[i]:
            fn = fn + 1

    p = tp / (tp + fp)
    r = tp / (tp + fn)

    return p, r, s

# Calculate precision and recall
pr = precision_recall(X_test, Y_test, optimum_beta)

# CrossValidation for learnng rate lr.

def CrossValidation(lr_list, X_train, Y_train, k_fold, param):  # K-fold CV.

    error_global = []  # For storing the errors of each (lr_list,structure_list) pair.

    for i in lr_list:
        error_list = []
        for k in range(1, k_fold):
            x_test = X_train[(k - 1) * int(len(X_train) / k_fold):k * int(len(X_train) / k_fold)]
            x_train = np.delete(X_train[0:int(len(X_train))],
                                range((k - 1) * int(len(X_train) / k_fold), k * int(len(X_train) / k_fold)),
                                axis=0)  # Assume 2000 train data.

            y_test = Y_train[(k - 1) * int(len(Y_train) / k_fold):k * int(len(Y_train) / k_fold)]
            y_train = np.delete(Y_train[0:int(len(Y_train))],
                                range((k - 1) * int(len(Y_train) / k_fold), k * int(len(Y_train) / k_fold)), axis=0)

            sol = GD(x_train, y_train, param, i, 2000)

            pred = score(x_test, sol[0])

            error_1 = error(pred, y_test)

            error_list.append(error_1)  #

        error_global.append(
            sum(error_list) / k_fold)  # Find the total error via sum function and divide it by the # of folds (i.e. k_fold) to find average error, then append it into the list.

    sorted_error = np.argsort(
        error_global)  # Sort the average CV errors for each (lr_list,structure_list) pair by their indices so that we can easily access the index of the optimum k value.

    min_index = sorted_error[0]

    min_lr = lr_list[min_index]

    return min_lr, error_global[min_index], error_global

lr_list = [i*0.05 for i in range(1,21)]
k_fold = 5
Validate = CrossValidation(lr_list, X_train, Y_train, k_fold, param)

fig, ax = plt.subplots()
plt.plot(lr_list, Validate[2], 'k-', label='optimum lr = {}'.format(Validate[0]))
ax.set_title('Lr vs Avarage test error')
ax.set_xlabel('Learning Rate (lr)')
ax.set_ylabel('Avarage Test Error')
ax.legend(shadow=True, fancybox=True)

