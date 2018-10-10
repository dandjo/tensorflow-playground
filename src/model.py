import tensorflow as tf

# To make a model trainable, we need to be able to modify the graph to get new
# outputs with the same input. Variables allow us to add trainable parameters
# to a graph.

# model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# inputs and outputs
in_ = tf.placeholder(tf.float32)
out_ = tf.placeholder(tf.float32)

# the model
linear_model = W * in_ + b

# loss
squared_delta = tf.square(linear_model - out_)
loss = tf.reduce_sum(squared_delta)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(loss, {
        in_: [1, 2, 3, 4],
        out_: [0, -1, -2, -3]
    }))  # 23.66

# optimize model
optimizer = tf.train.GradientDescentOptimizer(0.01)  # 0.01 steps to change var
train = optimizer.minimize(loss)

# run with variable range and optimizer
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train, {
            in_: [1, 2, 3, 4],
            out_: [0, -1, -2, -3]
        })
    # this will output the trained values of W and b
    print(sess.run([W, b]))  # [array([-0.9999969], dtype=float32),
    #                           array([0.9999908], dtype=float32)]
