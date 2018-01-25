"""
Example of cholesky decomposition:
https://en.wikipedia.org/wiki/Cholesky_decomposition
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()
sess = tf.Session()
x_values = np.linspace(0, 10, 100)
y_values = x_values + np.random.normal(0, 1, 100)
x_values_column = np.transpose(np.matrix(x_values))
ones_column = np.transpose(np.matrix(np.repeat(1, 100)))
A = np.column_stack((x_values_column, ones_column))
b = np.transpose(np.matrix(y_values))
A_tensor = tf.constant(A)
b_tensor = tf.constant(b)
tA_A = tf.matmul(tf.transpose(A_tensor), A)
L = tf.cholesky(tA_A)
tA_b = tf.matmul(tf.transpose(A_tensor), b)
pre_solution = tf.matrix_solve(L, tA_b)
solution = tf.matrix_solve(tf.transpose(L), pre_solution)
evaluation = sess.run(solution)
slope = evaluation[0][0]
y_intercept = evaluation[1][0]
fit = x_values*slope + y_intercept
MSE = np.mean((fit - x_values)**2)
# PRINT OUTPUT
print("Slope = " + str(slope))
print("Y intercept = " + str(y_intercept))
print("Mean Square Error = " + str(MSE))
# Plotting
plt.plot(x_values, y_values, "bo", label="data")
plt.plot(x_values, fit, "r-", label="fit")
plt.grid(True)
plt.show()
