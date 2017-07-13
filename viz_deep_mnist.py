import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# import matplotlib.pyplot as plt
import numpy as np
# import numpy.linalg as la
import time
import shutil
from subprocess import call

# TODO: This is using the old API. Will need to upgrade.


def summaries(var):
    mean = tf.reduce_mean(var)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary('mean', mean)
    tf.scalar_summary('stddev', stddev)
    tf.histogram_summary('hist', var)


def generate_data():
    return input_data.read_data_sets('/tmp/data', one_hot=True)


def hidden_layer(x_in, dim_in, dim_out, prefix='h'):
    '''
    Make layers
    '''
    with tf.name_scope(prefix + '_w'):
        w = tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=0.1))
        summaries(w)

    with tf.name_scope(prefix + '_b'):
        b = tf.Variable(tf.constant(0.1, shape=[dim_out]))
        summaries(b)

    with tf.name_scope(prefix + '_z'):
        z = tf.nn.relu(tf.matmul(x_in, w) + b)
        summaries(z)
    return z


def output_layer(x_in, dim_in, dim_out, prefix='o'):
    '''
    Make layers
    '''
    with tf.name_scope(prefix + '_w'):
        w = tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=0.1))
        summaries(w)

    with tf.name_scope(prefix + '_b'):
        b = tf.Variable(tf.constant(0.1, shape=[dim_out]))
        summaries(b)

    with tf.name_scope(prefix + '_y'):
        y = tf.matmul(x_in, w) + b
        summaries(y)
    return y


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
    LOGDIR = './graphs'
    shutil.rmtree(LOGDIR, ignore_errors=True)

    # Set up tf variables.
    x_p = tf.placeholder(shape=[None, NUM_PIXELS],
                         dtype=tf.float32,
                         name='x-input')
    y_p = tf.placeholder(shape=[None, NUM_CLASSES],
                         dtype=tf.float32,
                         name='y-input')
    # Layers
    z1 = hidden_layer(x_p, NUM_PIXELS, NUM_NL1, 'l1')
    z2 = hidden_layer(z1, NUM_NL1, NUM_NL2, 'l2')
    y = output_layer(z2, NUM_NL2, NUM_CLASSES, 'out')

    # Set up trainer.
    cross_entropy_reduced = []
    with tf.name_scope('loss'):
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_p,
                                                                logits=y)
        cross_entropy_reduced = tf.reduce_mean(cross_entropy)
        summaries(cross_entropy_reduced)

    train_step = optimizer.minimize(cross_entropy_reduced)

    # Initialize.
    sess.run(tf.initialize_all_variables())

    # Set up logger.
    train_writer = tf.train.SummaryWriter(LOGDIR)
    train_writer.add_graph(sess.graph)
    summary_op = tf.merge_all_summaries()
    # Check set up time.
    print("Time to set up = {} seconds".format(time.time() - start_time))
    start_time = time.time()

    # Set up previous weights for convergence test.
    # w_o = float("inf")
    # b_o = float("inf")

    for t in range(EPOCHS):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        summary_result, _ = sess.run([summary_op, train_step],
                                     feed_dict={x_p: batch_xs, y_p: batch_ys})
        train_writer.add_summary(summary_result, t)
        if t % 100 == 0:
            print('train_step = {}'.format(t))

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_p, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', accuracy)

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

    call(["tensorboard", "--logdir", LOGDIR])
if __name__ == '__main__':
    main()
