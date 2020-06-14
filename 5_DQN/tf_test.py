import tensorflow as tf
import numpy as np
import matplotlib as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size])+0.1, name='b')
        with tf.name_scope('biases'):
            Wx_puls_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_puls_b
        else:
            outputs = activation_function(Wx_puls_b)
        return outputs

x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
print(x_data.dtype)
nose = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data)-0.5+nose

with tf.name_scope('inputs'):
    ys = tf.placeholder(tf.float32, [None, 1], name='x_input')
    xs = tf.placeholder(tf.float32, [None, 1], name='y_input')

L1 = add_layer(x_data, 1, 10, activation_function = tf.nn.relu)
prediction = add_layer(L1, 10, 1, activation_function = None)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_data-prediction),
                     reduction_indices=[1]))
with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
writer = tf.summary.FileWriter("E:\RL", sess.graph)
#
# for i in range(1000):
#     sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
#     if i % 50:
#         print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))

