# Libraries
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functions import smo_algorithm
from functions import RBF_Approx


# Clean breast cancer data
# ------------------------------------------------------------------------------

print("------------- Clean breast cancer data -------------")
print("")

data = pd.read_csv('../datasets/breast_cancer.csv')

# data X (consists of all but first two and very last two columns)
indicators = data.iloc[:, 2:-3].to_numpy()

# data y (classifiers / diagnoses)
diagnosis = data['diagnosis'].to_numpy()

def string_to_binary_classifiers(diagnoses):
    """
    convert diagnosis classifiers to binary classifiers

    Args:
        diagnosis: List[String]

    Returns:
        null
    """
    for i in range(0, len(diagnoses)):
        if diagnoses[i] == 'B':
            diagnoses[i] = -1.0
        elif diagnoses[i] == 'M':
            diagnoses[i] = 1.0

# convert diagnosis classifiers to binary classifiers
string_to_binary_classifiers(diagnosis)


# Linear SVM for Breast Cancer data
# ------------------------------------------------------------------------------

print("------------- Linear SVM for Breast Cancer Data -------------")
print("")

# initialize lists to hold trained model values w, b and success rates
Trained_models = []
Success_rates = []

# split input and output data into 10 parts
split_indicators = np.array_split(indicators, 10)
split_diagnoses = np.array_split(diagnosis, 10)

# set params C, max iterations, and threshold
C = 1.0
max_iter = 500
thresh = 1e-300

# for each subset get optimal w, b and append to trained_models list
for n in range(0, 10):
    alph, b = smo_algorithm(split_indicators[n], split_diagnoses[n], C, max_iter, thresh)

    w = 0
    for i in range(0, len(alph)):
        a = alph[i]
        y = split_diagnoses[n][i]
        x = split_indicators[n][i]
        ayx = a * y * x
        w += ayx

    Trained_models.append([w, b])

# initiate variables to keep track of number of
# successfully classified points and total tested points
total = 0
success = 0

# test each point for successful classification
for n in range(0, 10):

    for i in range(0, n):
        for j in range(0, len(split_indicators[i])):
            x = split_indicators[i][j]
            y = split_diagnoses[i][j]
            w = Trained_models[i][0]
            w_transpose = np.transpose(w)
            b = Trained_models[i][1]
            check = y * (w_transpose.dot(x) + b) - 1
            # if successfully classified add one to success
            if check >= 0:
                success += 1
            # always add one to total
            total += 1

    for i in range(n+1, 10):
        for j in range(0, len(split_indicators[i])):
            x = split_indicators[i][j]
            y = split_diagnoses[i][j]
            w = Trained_models[i][0]
            w_transpose = np.transpose(w)
            b = Trained_models[i][1]
            check = y * (w_transpose.dot(x) + b) - 1
            # if successfully classified add one to success
            if check >= 0:
                success += 1
            # always add one to total
            total += 1

    # calculate percentage of successfully classified points
    # and append it to success_rates
    percent_success = success/total
    Success_rates.append(percent_success)

    print("Success for n={}: {}".format(n, percent_success))
    print("")

# plot histogram of success rates
plt.hist(Success_rates, density=True, bins=50)
plt.ylabel('Distribution')
plt.xlabel('Success Rate')

plt.show()


# RBF SVM for Breast Cancer data
# ------------------------------------------------------------------------------

print("------------- RBF SVM for Breast Cancer Data -------------")
print("")

# set parameter values
gamma = 1e-6

deg2 = 2
deg3 = 3

# employ RBF approximation

# initiate arrays to hold trained models and success rates for RBF approx X values
# for degree 2, 3
Trained_models2 = []
Trained_models3 = []

Success_rates2 = []
Success_rates3 = []

# initiate arrays to hold X values for degree 2, 3
X2_list = []
X3_list = []

# for each subset get optimal w, b and append to rbf_trained_models list
for n in range(0, 10):
    # get RBF approx X for degree 2, 3
    X2 = RBF_Approx(split_indicators[n], gamma, deg2)
    X3 = RBF_Approx(split_indicators[n], gamma, deg3)

    # append to X value lists
    X2_list.append(X2)
    X3_list.append(X3)

    # get alpha and b for X2, X3
    alph2, b2 = smo_algorithm(X2, split_diagnoses[n], C, max_iter, thresh)
    alph3, b3 = smo_algorithm(X3, split_diagnoses[n], C, max_iter, thresh)

    # get w for X2, X3
    w2 = 0
    for i in range(0, len(alph2)):
        a = alph2[i]
        y = split_diagnoses[n][i]
        x = X2[n][i]
        ayx = a * y * x
        w2 += ayx

    Trained_models2.append([w2, b2])

    w3 = 0
    for i in range(0, len(alph3)):
        a = alph3[i]
        y = split_diagnoses[n][i]
        x = X3[n][i]
        ayx = a * y * x
        w3 += ayx

    Trained_models3.append([w3, b3])

# initiate variables to keep track of number of
# successfully classified points and total tested points
# for degree 2, 3 RBF approximations
total2 = 0
total3 = 0

success2 = 0
success3 = 0

# test each point for successful classification
for n in range(0, 10):

    for i in range(0, n):
        for j in range(0, len(X2_list[i])):
            x = X2_list[i][j]
            y = split_diagnoses[i][j]
            w = Trained_models2[i][0]
            w_transpose = np.transpose(w)
            b = Trained_models2[i][1]
            check = y * (w_transpose.dot(x) + b) - 1
            # if successfully classified add one to success
            if check >= 0:
                success2 += 1
            # always add one to total
            total2 += 1

    for i in range(0, n):
        for j in range(0, len(X3_list[i])):
            x = X3_list[i][j]
            y = split_diagnoses[i][j]
            w = Trained_models3[i][0]
            w_transpose = np.transpose(w)
            b = Trained_models3[i][1]
            check = y * (w_transpose.dot(x) + b) - 1
            # if successfully classified add one to success
            if check >= 0:
                success3 += 1
            # always add one to total
            total3 += 1

    for i in range(n+1, 10):
        for j in range(0, len(X2_list[i])):
            x = X2_list[i][j]
            y = split_diagnoses[i][j]
            w = Trained_models2[i][0]
            w_transpose = np.transpose(w)
            b = Trained_models2[i][1]
            check = y * (w_transpose.dot(x) + b) - 1
            # if successfully classified add one to success
            if check >= 0:
                success2 += 1
            # always add one to total
            total2 += 1

    for i in range(n+1, 10):
        for j in range(0, len(X3_list[i])):
            x = X3_list[i][j]
            y = split_diagnoses[i][j]
            w = Trained_models3[i][0]
            w_transpose = np.transpose(w)
            b = Trained_models3[i][1]
            check = y * (w_transpose.dot(x) + b) - 1
            # if successfully classified add one to success
            if check >= 0:
                success3 += 1
            # always add one to total
            total3 += 1

    # calculate percentage of successfully classified points
    # and append it to success_rates
    percent_success2 = success2/total2
    percent_success3 = success3/total3
    Success_rates2.append(percent_success2)
    Success_rates3.append(percent_success3)

    print("Success for degree = {}, n={}: {}".format(2, n, percent_success2))
    print("Success for degree = {}, n={}: {}".format(3, n, percent_success3))
    print("")
