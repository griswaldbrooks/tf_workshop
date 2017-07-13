import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# import matplotlib.pyplot as plt
import numpy as np
# import numpy.linalg as la
import time


def generate_data():
    return input_data.read_data_sets('/tmp/data', one_hot=True)


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
    BATCH_SIZE = 100
    LEARNING_RATE = 0.5
    EPOCHS = 2000

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    # Set up tf variables.
    x_p = tf.placeholder(shape=[None, NUM_PIXELS],
                         dtype=tf.float32,
                         name='x-input')
    y_p = tf.placeholder(shape=[None, NUM_CLASSES],
                         dtype=tf.float32,
                         name='y-input')
    w = tf.Variable(tf.truncated_normal([NUM_PIXELS, NUM_CLASSES], stddev=0.1),
                    name='w')
    b = tf.Variable(tf.zeros([NUM_CLASSES]), name='b')
    y = tf.matmul(x_p, w) + b

    # Set up trainer.
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_p,
                                                            logits=y)
    cross_entropy_reduced = tf.reduce_mean(cross_entropy)
    train_step = optimizer.minimize(cross_entropy_reduced)

    # Initialize.
    sess.run(tf.initialize_all_variables())

    # Check set up time.
    print("Time to set up = {} seconds".format(time.time() - start_time))
    start_time = time.time()

    # Set up previous weights for convergence test.
    # w_o = float("inf")
    # b_o = float("inf")

    for t in range(EPOCHS):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train_step, feed_dict={x_p: batch_xs, y_p: batch_ys})
        if t % 100 == 0:
            print('train_step = {}'.format(t))

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_p, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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
