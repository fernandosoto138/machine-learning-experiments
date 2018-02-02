import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

# Training data generation
x_vals = np.linspace(0, 10, 100)
y_vals = x_vals + np.random.normal(0, 1, 100)
x_vals_column = np.transpose(np.matrix(x_vals))
ones_column = np.transpose(np.matrix(np.repeat(1, 100)))
A = np.column_stack((x_vals_column, ones_column))
b = np.transpose(np.matrix(y_vals))
# Constants
dim_x = 100
learning_rate = 0.001
batch_size = 25
training_epochs = 101
display_step = 25
# Tensorflow model:
graph = tf.Graph()
sess = tf.Session(graph=graph)
with graph.as_default():
    with tf.name_scope("Inputs"):
        x_data = tf.placeholder(dtype=tf.float32, name="x_data")
        y_data = tf.placeholder(dtype=tf.float32, name="y_data")
    with tf.name_scope("Variables"):
        m_tensor = tf.Variable(np.random.randn(), name="slope")
        b_tensor = tf.Variable(np.random.randn(), name="yIntercept")
        asdasdasd = "a"
    with tf.name_scope("LinearModel"): # y=m*x+b
        output_y = m_tensor*x_data + b_tensor
    with tf.name_scope("LossFunction"):
        loss = tf.reduce_mean(tf.square(output_y - y_data))
        variable_summaries(loss)
    with tf.name_scope("Optimizer"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    with tf.name_scope("Training"):
        init = tf.global_variables_initializer()
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./output", sess.graph)
        sess.run(init)
        for epoch in range(training_epochs):
            for (x, y) in zip(x_vals, y_vals):
                sess.run(optimizer, feed_dict={x_data: x, y_data: y})
            summary, loss_value = sess.run([merged, loss], feed_dict={x_data: x_vals, y_data: y_vals})
            writer.add_summary(summary, epoch)
            if (epoch+1) % display_step == 0:
                print("Epoch:"+str(epoch)+" loss:"+str(loss_value))
        training_loss = sess.run(loss, feed_dict={x_data: x_vals, y_data: y_vals})
       
        #print(training_loss)
        slope = sess.run(m_tensor)
        y_intercept = sess.run(b_tensor)
        print("Slope = " + str(slope))
        print("Y intercept = " + str(y_intercept))
        print("Loss = " + str(training_loss))
        
        
        writer.close()
   


