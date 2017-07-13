import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import time


def generate_data(n=100, m=0.01, b=0.3):
    # create some data using numpy. y = x * 0.1 + 0.3 + noise
    x = np.random.rand(n).astype(np.float32)
    noise = np.random.normal(scale=0.1, size=len(x))
    y = m * x + b + noise
    return (x, y)


def main():
    # Make training data
    x_d, y_d = generate_data(m=2)

    # Get start time for profiling.
    start_time = time.time()

    # Initalize tf
    tf.reset_default_graph()
    sess = tf.Session()

    # Set up tf variables.
    x_p = tf.placeholder(shape=[None], dtype=tf.float32, name='x-input')
    y_p = tf.placeholder(shape=[None], dtype=tf.float32, name='y-input')
    w = tf.Variable(tf.random_normal([1]), name='w')
    b = tf.Variable(tf.random_normal([1]), name='b')
    y = w * x_p + b

    # Set up trainer.
    LEARNING_RATE = 0.5
    loss = tf.reduce_mean(tf.square(y - y_p))
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    train = optimizer.minimize(loss)

    # Initialize.
    sess.run(tf.initialize_all_variables())

    # Check set up time.
    print("Set up time = {} seconds".format(time.time() - start_time))
    start_time = time.time()

    # Set up previous weights for convergence test.
    w_o = float("inf")
    b_o = float("inf")

    # Train model.
    for steps in range(201):
        sess.run([train], feed_dict={x_p: x_d, y_p: y_d})
        w_p, b_p = sess.run([w, b])

        if steps % 20 == 0:
            print("{} w = {} b = {}".format(steps, w_p, b_p))

        # Test for convergence
        s_error = la.norm(np.array([w_p, b_p]) - np.array([w_o, b_o]))
        if s_error < 1e-4:
            break

        # Update previous.
        w_o = w_p
        b_o = b_p

    # Print results.
    print("Training time = {} seconds".format(time.time() - start_time))
    wval = sess.run(w)
    bval = sess.run(b)
    print ("w: %f, b: %f" % (wval, bval))

    # Plot data and model.
    plt.plot(x_d, y_d, 'b.')
    x = np.linspace(0, 1, 100)
    y = x * wval + bval
    plt.plot(x, y, 'r.')

    plt.show()

if __name__ == '__main__':
    main()
