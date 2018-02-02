import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
# Prepare the data
x_vals = np.linspace(0, 10, 100)
y_vals = x_vals + np.random.normal(0, 1, 100)
x_vals_column = np.transpose(np.matrix(x_vals))
ones_column = np.transpose(np.matrix(np.repeat(1, 100)))
A = np.column_stack((x_vals_column, ones_column))
b = np.transpose(np.matrix(y_vals))
# Configure tensors
graph = tf.Graph()
with graph.as_default():
    with tf.name_scope("Input"):
        # A_tensor = tf.constant(A, name="A")
        # b_tensor = tf.constant(b, name="b")
        A_tensor = tf.placeholder(dtype=tf.float32, shape=[100, 2], name="A")
        b_tensor = tf.placeholder(dtype=tf.float32, shape=[100, 1], name="b")
    with tf.name_scope("Transformation"):
        tA = tf.transpose(A_tensor)
        tA_A = tf.matmul(tA, A_tensor)
        tA_A_inv = tf.matrix_inverse(tA_A)
        product = tf.matmul(tA_A_inv, tA)
    with tf.name_scope("Output"):
        solution = tf.matmul(product, b_tensor)
    with tf.name_scope("Summaries"):
        tf.summary.tensor_summary("Output_Summary", solution)
    with tf.name_scope("global_ops"):
        merged_summaries = tf.summary.merge_all()
# Start session in tensorflow
sess = tf.Session(graph=graph)
# Start summary writer for tensorboard 
writer = tf.summary.FileWriter("./output", sess.graph)
feed = {A_tensor: A, b_tensor: b}
solution_eval, summaries = sess.run([solution, merged_summaries], feed_dict=feed)
writer.close()
# slope = pendiente
slope = solution_eval[0][0]
y_intercept = solution_eval[1][0]
best_fit = []
for i in x_vals:
    val = slope*i+y_intercept
    best_fit.append(val)
MSE = ((np.array(best_fit) - x_vals)**2).mean()
# plt.plot(x_vals, y_vals, 'o', label='Data')
# plt.plot(x_vals, best_fit, 'r-', label='Best fit line')
# plt.plot(x_vals, x_vals, 'b-', label='Best fit line')
# plt.legend(loc='upper left')
# plt.grid(True)
# plt.show()
print("Slope = " + str(slope))
print("Y intercept = " + str(y_intercept))
print("Mean Square Error = " + str(MSE))
