# Libraries
# ------------------------------------------------------------------------------
from collections import Counter
from itertools import combinations
import numpy as np
import scipy.linalg as LA


# Functions
# ------------------------------------------------------------------------------
def fact(n):
    """Factorial of an integer n>=0."""
    if n in [0, 1]:
        return 1
    else:
        return n * fact(n - 1)


def partition(number: int, max_vals: tuple):
    S = set(combinations((k for i, val in enumerate(max_vals) for k in [i] * val), number))
    for s in S:
        c = Counter(s)
        yield tuple([c[n] for n in range(len(max_vals))])


def RBF_Approx(X, gamma, deg):
    """Transforms data in X to its RBF representation, but as an approximation
    in deg degrees.  gamma = 1/2."""
    new_X = []
    N = X.shape[0]
    n = X.shape[1]
    count = 0
    for i in range(N):
        vec = []
        for k in range(deg + 1):
            if k == 0:
                vec += [1]
            else:
                tup = (k,) * n
                parts = list(partition(k, tup))
                for part in parts:
                    vec += [np.prod(
                        [np.sqrt(gamma ** deg) * (X[i, s] ** part[s]) / np.sqrt(fact(part[s])) for s in range(n)])]
        new_X += [np.exp(-gamma * LA.norm(X[i, :]) ** 2) * np.asarray(vec)]
        print(str(count) + " of " + str(N))
        count += 1

    return np.asarray(new_X)


def smo_algorithm(X, y, C, max_iter, thresh):
    """Optimizes Lagrange multipliers in the dual formulation of SVM.
        X: The data set of size Nxn where N is the number of observations and
           n is the length of each feature vector.
        y: The class labels with values +/-1 corresponding to the feature vectors.
        C: A threshold positive value for the size of each lagrange multiplier.
           In other words 0<= a_i <= C for each i.
        max_iter: The maximum number of successive iterations to attempt when
                  updating the multipliers.  The multipliers are randomly selected
                  as pairs a_i and a_j at each iteration and updates these according
                  to a systematic procedure of thresholding and various checks.
                  A counter is incremented if an update is less than the value
                  thresh from its previous iteration.  max_iter is the maximum
                  value this counter attains before the algorithm terminates.
        thresh: The minimum threshold difference between an update to a multiplier
                and its previous iteration.
    """
    alph = np.zeros(len(y))
    b = 0
    count = 0
    while count < max_iter:

        num_changes = 0

        for i in range(len(y)):
            w = np.dot(alph * y, X)
            E_i = np.dot(w, X[i, :]) + b - y[i]

            if (y[i] * E_i < -thresh and alph[i] < C) or (y[i] * E_i > thresh and alph[i] > 0):
                j = np.random.choice([m for m in range(len(y)) if m != i])
                E_j = np.dot(w, X[j, :]) + b - y[j]

                a_1old = alph[i]
                a_2old = alph[j]
                y_1 = y[i]
                y_2 = y[j]

                # Compute L and H
                if y_1 != y_2:
                    L = np.max([0, a_2old - a_1old])
                    H = np.min([C, C + a_2old - a_1old])
                elif y_1 == y_2:
                    L = np.max([0, a_1old + a_2old - C])
                    H = np.min([C, a_1old + a_2old])

                if L == H:
                    continue
                eta = 2 * np.dot(X[i, :], X[j, :]) - LA.norm(X[i, :]) ** 2 - LA.norm(X[j, :]) ** 2
                if eta >= 0:
                    continue
                # Clip value of a_2
                a_2new = a_2old - y_2 * (E_i - E_j) / eta
                if a_2new >= H:
                    a_2new = H
                elif a_2new < L:
                    a_2new = L

                if abs(a_2new - a_2old) < thresh:
                    continue

                a_1new = a_1old + y_1 * y_2 * (a_2old - a_2new)

                # Compute b
                b_1 = b - E_i - y_1 * (a_1new - a_1old) * LA.norm(X[i, :]) - y_2 * (a_2new - a_2old) * np.dot(X[i, :],
                                                                                                              X[j, :])
                b_2 = b - E_j - y_1 * (a_1new - a_1old) * np.dot(X[i, :], X[j, :]) - y_2 * (a_2new - a_2old) * LA.norm(
                    X[j, :])

                if 0 < a_1new < C:
                    b = b_1
                elif 0 < a_2new < C:
                    b = b_2
                else:
                    b = (b_1 + b_2) / 2

                num_changes += 1
                alph[i] = a_1new
                alph[j] = a_2new

        if num_changes == 0:
            count += 1
        else:
            count += 0
        #print(count)
    return alph, b

def hinge_loss(X, y, w, b):
    """Here X is assumed to be Nxn where each row is a data point of length n and
    N is the number of data points.  y is the vector of class labels with values
    either +1 or -1.  w is the support vector and b the corresponding bias."""
    # get total number of datapoints
    N = len(X)

    # initialize list to hold f_i values
    f_list = []

    count = 0

    # get f_i
    for n in range(0, N):
        x_i = X[count]
        y_i = y[count]
        distance = 1 - y_i * ((w.dot(x_i)) + b)
        f_list.append(max(0, distance))

        count += 1

    # take mean of f_list to get f(w, b)
    f = 0.5 * LA.norm(w)**2 + np.mean(f_list)
    return f


def hinge_deriv(X, y, w, b):
    """Here X is assumed to be Nxn where each row is a data point of length n and
    N is the number of data points.  y is the vector of class labels with values
    either +1 or -1.  w is the support vector and b the corresponding bias."""
    # get total number of datapoints
    N = len(X)

    # initialize list to hold df_i_dw (derivative of f_i wrt w) values
    df_w_list = []
    # initialize list to hold df_i_db (derivative of f_i wrt b) values
    df_b_list = []

    for n in range(0, N):
        x_i = X[n]
        y_i = y[n]

        distance = y_i * ((w.dot(x_i)) + b)
        if distance >= 1:
            df_i_dw = w
            df_i_db = 0
        else:
            df_i_dw = w - (y_i * x_i)
            df_i_db = - y_i

        df_w_list.append(df_i_dw)
        df_b_list.append(df_i_db)

    df_dw = np.mean(df_w_list)
    df_db = np.mean(df_b_list)

    return df_dw, df_db


def hinge_gradient_descent(x, y, w, b, K=0.01, max_iter=10000):
    eps = 0.1

    dw, db = hinge_deriv(x, y, w, b)

    iter = 0

    while (iter < max_iter) and (np.sqrt(LA.norm(dw)**2 + LA.norm(db)**2) > K):
        dw, db = hinge_deriv(x, y, w, b)
        w = w - (eps * dw)
        b = b - (eps * db)

        iter += 0

    return w, b


def radial_kernel(x, y):
    z = np.sqrt(x**2 + y**2)
    return z