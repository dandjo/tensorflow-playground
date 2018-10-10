import tensorflow as tf

# A graph can be parameterized to accept external inputs, known as placeholders.
# A placeholder is a promise to provide a value later.

node1 = tf.placeholder(tf.float32)
node2 = tf.placeholder(tf.float32)

print(node1)  # Tensor("Placeholder:0", dtype=float32)
print(node2)  # Tensor("Placeholder_1:0", dtype=float32)

adder_node = node1 + node2

print(adder_node)  # Tensor("add:0", dtype=float32)

with tf.Session() as sess:
    output = sess.run(adder_node, {node1: [1, 3], node2: [2, 4]})
    print(output)  # [3. 7.]
