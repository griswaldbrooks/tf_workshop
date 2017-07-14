import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time


def generate_data():
    return input_data.read_data_sets('/tmp/data', one_hot=True)


def cnn_layer(x_in, w_shape, b_shape):
    '''
    Make layers
    '''
    w = tf.Variable(tf.truncated_normal(w_shape, stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=b_shape))
    stride_conv = [1, 1, 1, 1]
    stride_pool = [1, 2, 2, 1]
    k_size_pool = [1, 2, 2, 1]
    c = tf.nn.conv2d(x_in, w, strides=stride_conv, padding='SAME')
    h = tf.nn.relu(c + b)
    return tf.nn.max_pool(h, ksize=k_size_pool, strides=stride_pool, padding='SAME')


def fc_layer(x_in, dim_in, dim_out):
    '''
    Make layers
    '''
    w = tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[dim_out]))
    return tf.nn.relu(tf.matmul(x_in, w) + b)


def output_layer(x_in, dim_in, dim_out):
    '''
    Make layers
    '''
    w = tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[dim_out]))
    return tf.matmul(x_in, w) + b


def main():
    # Get start time for profiling.
    start_time = time.time()

    # Make training data
    mnist = generate_data()
    print ('Train, validation, test: {}, {}, {}'.format(
        len(mnist.train.images),
        len(mnist.validation.images),
        len(mnist.test.images)))

    # 2. The format of the labels is 'one-hot'.
    # The fifth image happens to be a '1'.
    # This is represented as '[ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]'
    print (mnist.train.labels[4])

    # You can find the index of the label, like this:
    print (np.argmax(mnist.train.labels[4]))

    # 3. An image is a 'flattened' array of 28*28 = 784 pixels.
    print (len(mnist.train.images[4]))

    # Check set up time.
    print("Time to get data = {} seconds".format(time.time() - start_time))
    start_time = time.time()

    # Initalize tf
    NUM_CLASSES = 10
    NUM_PIXELS = 28 * 28
    BATCH_SIZE = 200
    LEARNING_RATE = 0.001
    EPOCHS = 4000
    NUM_NL1 = 100
    NUM_NL2 = 100

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    # Set up tf variables.
    x_p = tf.placeholder(shape=[None, NUM_PIXELS],
                         dtype=tf.float32,
                         name='x-input')
    y_p = tf.placeholder(shape=[None, NUM_CLASSES],
                         dtype=tf.float32,
                         name='y-input')
    # Layers
    x_img = tf.reshape(x_p, [-1, 28, 28, 1])
    z1 = cnn_layer(x_img, [5, 5, 1, 32], [32])
    z2 = cnn_layer(z1, [5, 5, 32, 64], [64])
    z2_flat = tf.reshape(z2, [-1, 7 * 7 * 64])
    z3 = fc_layer(z2_flat, 7 * 7 * 64, 1024)
    keep_prob = tf.placeholder(tf.float32)
    z3_dropped = tf.nn.dropout(z3, keep_prob)
    y = output_layer(z3_dropped, 1024, NUM_CLASSES)

    # Set up trainer.
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_p,
                                                            logits=y)
    cross_entropy_reduced = tf.reduce_mean(cross_entropy)
    train_step = optimizer.minimize(cross_entropy_reduced)

    # Initialize.
    sess.run(tf.initialize_all_variables())

    # Check set up time.
    print("Time to set up = {} seconds".format(time.time() - start_time))
    start_time = time.time()

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_p, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    for t in range(EPOCHS):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        if t % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x_p: batch_xs, y_p: batch_ys, keep_prob: 1.0})
            print('train_step = {} accuracy = {}'.format(t, train_accuracy))

        train_step.run(feed_dict={x_p: batch_xs, y_p: batch_ys, keep_prob: 0.5})

    # Print results.
    print("Training time = {} seconds".format(time.time() - start_time))
    start_time = time.time()

    print(sess.run(accuracy, feed_dict={x_p: mnist.train.images,
                                        y_p: mnist.train.labels}))
    print("Predict time train = {} seconds".format(time.time() - start_time))
    print("{} images".format(len(mnist.train.images)))
    start_time = time.time()

    print(sess.run(accuracy, feed_dict={x_p: mnist.test.images,
                                        y_p: mnist.test.labels}))
    print("Predict time test = {} seconds".format(time.time() - start_time))
    print("{} images".format(len(mnist.test.images)))

if __name__ == '__main__':
    main()
