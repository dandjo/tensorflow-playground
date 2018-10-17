import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)

print(node1)  # Tensor("Const:0", shape=(), dtype=float32)
print(node2)  # Tensor("Const_1:0", shape=(), dtype=float32)

node3 = node1 * node2

print(node3)  # Tensor("mul:0", shape=(), dtype=float32)

with tf.Session() as sess:
    output = sess.run([node1, node2])
    print(output)  # [3.0, 4.0]

with tf.Session() as sess:
    output = sess.run(node3)
    print(output)  # 12
    file_writer = tf.summary.FileWriter('graph', sess.graph)
    # run `tensorboard --logdir="graph"` in command line to show the result
