import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
# Start session in tensorflow
sess = tf.Session()
x_vals = np.linspace(0, 10, 100)
y_vals = x_vals + np.random.normal(0, 1, 100)
x_vals_column = np.transpose(np.matrix(x_vals))
ones_column = np.transpose(np.matrix(np.repeat(1, 100)))
A = np.column_stack((x_vals_column, ones_column))
b = np.transpose(np.matrix(y_vals))
A_tensor = tf.constant(A)
b_tensor = tf.constant(b)
tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
tA_A_inv = tf.matrix_inverse(tA_A)
product = tf.matmul(tA_A_inv, tf.transpose(A_tensor))
solution = tf.matmul(product, b_tensor)
solution_eval = sess.run(solution)
# slope = pendiente
slope = solution_eval[0][0]
y_intercept = solution_eval[1][0]
best_fit = []
for i in x_vals:
    val = slope*i+y_intercept
    best_fit.append(val)
MSE = ((np.array(best_fit) - x_vals)**2).mean()
plt.plot(x_vals, y_vals, 'o', label='Data')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line')
plt.plot(x_vals, x_vals, 'b-', label='Best fit line')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()
# Print data
print("Slope = " + str(slope))
print("Y intercept = " + str(y_intercept))
print("Mean Square Error = " + str(MSE))
