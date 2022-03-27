# Libraries
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functions import hinge_gradient_descent
from functions import smo_algorithm
from functions import radial_kernel


# SVM using hinge loss
# ------------------------------------------------------------------------------

print("------------- SVM using hinge loss -------------")
print("")

svm_data = pd.read_csv("../datasets/svm_test_2.csv")

input = svm_data.iloc[: , :2]
input_data = input.to_numpy()

output = svm_data.iloc[: , -1]
output_data = output.to_numpy()

x_coords = svm_data["0"].to_numpy()
y_coords = svm_data["1"].to_numpy()

w = np.array([-10, 10])
b = 5

optimal_w, optimal_b = hinge_gradient_descent(input_data, output_data, w, b)

print("optimal w: {}".format(optimal_w))
print("optimal b: {}".format(optimal_b))

print("")


# SVM using Lagrange multipliers
# ------------------------------------------------------------------------------

print("------------- SVM using Lagrange multipliers -------------")
print("")

max_iter = 2500
thresh = 1e-5

C_vals = [0.25, 0.5, 0.75, 1]

w_values = []
b_values = []

for C in C_vals:
    alph, b = smo_algorithm(input_data, output_data, C, max_iter, thresh)
    w0 = 0
    w1 = 0
    for i in range(0, len(alph)-1):
        a = alph[i]
        x = input_data[i]
        y = output_data[i]
        ayx = a * y * x
        w0 += ayx[0]
        w1 += ayx[1]
    w = np.array([w0, w1])
    w_values.append(w)
    b_values.append(b)
    print("optimal w for C={}: {}".format(C, w))
    print("optimal b for C={}: {}".format(C, b))
    print("")

print("")


# Plot SVM results
# ------------------------------------------------------------------------------

print("------------- Plot classic SVM results -------------")
print("")

# plot original datapoints
plt.scatter(x_coords, y_coords)

# plot hinge loss dividing line
x = np.linspace(-5, 5, 100)
y_class = - (1/(-2.31))*(-1.45*x + 5.55)
plt.plot(x, y_class, label='hinge loss (goal)')

for i in range(0, len(w_values)):
    w = w_values[i]
    b = b_values[i]
    C = C_vals[i]
    y_lagrange = -(1/w[1]) * (w[0] * x + b)
    plt.plot(x, y_lagrange, label='Lagrange for C={}'.format(C))

plt.legend()
plt.show()


# SVM for radial data with kernel embedding
# ------------------------------------------------------------------------------

print("------------- SVM for radial data using kernel embedding -------------")
print("")

# initiate list to hold corresponding z values
z_values = []

for pair in input_data:
    x = pair[0]
    y = pair[1]

    z = radial_kernel(x, y)
    z_values.append(z)

z_mat = []
while z_values != []:
    z_mat.append(z_values[:1])
    z_values = z_values[1:]
print(z_mat)

z_dim = np.asarray(z_mat)

# insert z values as third dimension to input data
# for kernel embedding
radial_input_data = np.insert(input_data, z_dim, axis=1)

# set parameters
C = 1.0
max_iter = 500
thresh = 1e-5

# initiate lists to hold w, b values
rad_b_values = []
rad_w_values = []

# perform smo algorithm and get w, b values for kernel embedded data
rad_alph, rad_b = smo_algorithm(radial_input_data, output_data, C, max_iter, thresh)
w0 = 0
w1 = 0
for i in range(0, len(rad_alph)-1):
    a = rad_alph[i]
    x = radial_input_data[i]
    y = output_data[i]
    ayx = a * y * x
    w0 += ayx[0]
    w1 += ayx[1]
rad_w = np.array([w0, w1])
rad_w_values.append(rad_w)
rad_b_values.append(rad_b)
print("optimal w for C={}: {}".format(C, rad_w))
print("optimal b for C={}: {}".format(C, rad_b))
print("")
