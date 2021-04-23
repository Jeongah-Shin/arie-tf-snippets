import tensorflow as tf

w1 = tf.Varaible(tf.random_normal(shape=[2]), name='w1')
w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Save model after 1000 iterations
saver.save(sess, 'my_model', global_step=1000)
# Except meta file, just restore the weights, biases, gradients
saver.save(sess, 'my_model', global_step=1000, write_meta_graph=False)
