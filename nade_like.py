'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf
from chord2vec.linear_models import data_processing as dp
import pickle
import numpy as np
import random

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_math_ops

# train_set = [[[0.0, 0.0, 0.0, 1,1, 0, 1, 0], [1, 1, 1, 1,0, 0, 1, 1], [1, 1, 1, 0,0, 1, 0, 1], [0, 0, 0, 1,0, 0, 0, 1]], \
#             [[1, 1, 0, 1,0, 1, 1, 0], [0, 0, 1, 1,1, 1, 1, 1], [0, 0, 1, 0,1, 0, 0, 1], [1, 1, 0, 1,1, 1, 0,1]]]
#
# test_set = [[ [1, 1, 1, 1,0, 0, 1, 1], [1, 1, 1, 0,0, 1, 0, 1], [0, 0, 0, 1,0, 0, 0, 1], [1, 0, 0, 0,1, 0, 0, 1]], \
#            [ [0, 0, 1, 1,1, 1, 1, 1], [0, 0, 1, 0,1, 0, 0, 1], [1, 1, 0, 1,1, 1, 0,1], [0, 1, 0, 0,0, 1, 0, 1]]]
# data_size = len(train_set[0])
# Parameters
learning_rate = 0.001
training_epochs = 500
batch_size = 128
display_step = 1

# Network Parameters
D = 1024
NUM_NOTES = 88
# tf Graph input
input = tf.placeholder("float", [None, NUM_NOTES])
target = tf.placeholder("float", [None, NUM_NOTES])

# Store layers weight & bias
weights = {
    'M1': tf.Variable(tf.random_normal([NUM_NOTES, D])),
    'M2': tf.Variable(tf.random_normal([D,NUM_NOTES])),
    # 'hidden2': tf.Variable(tf.random_normal([D, NUM_NOTES])),
    # 'pad' : tf.Variable(tf.constant(ones_triangular(NUM_NOTES).tolist()),trainable=False),
    'W': tf.Variable(tf.random_normal([D, NUM_NOTES]))
}

bias = {
    #     'hidden1': tf.Variable(tf.random_normal([D])),
    'M2': tf.Variable(tf.random_normal([D])),
    #     'out': tf.Variable(tf.random_normal([NUM_NOTES]))
}
def ones_triangular(dim):
    num_units = dim
    padding = np.zeros((num_units,num_units), np.float32)
    for i in range(num_units):
        for j in range(num_units):
            if i < j:
                padding[i][j] = 1.0
    return padding

def extend_vector(input,r,batch_size):
    """
    [a,b,c] --> [[a,a,a],[b,b,b],[c,c,c]] if D=3
    """
    #return list(map(lambda x: tf.matmul([[1.]]*r,[x]), input))
    return tf.batch_matmul(tf.ones([batch_size, r, 1]), tf.expand_dims(input, 1))

def mask(input, W, r=D):
    inputs = extend_vector(input,r,batch_size)
    return  tf.squeeze(tf.mul(inputs, [W]))
    #return list(map(lambda x: tf.mul(x,M), inputs))

def cumsum_weights(input, W, r=D):
    masked=mask(input,W,r)
    triangle = ones_triangular(NUM_NOTES)
    size = batch_size
    #return list(map(lambda x: tf.matmul(x, triangle), masked))
    return tf.batch_matmul(masked, np.array([triangle]*size))



def normalize(input):
    return tf.truediv(input, tf.maximum(1.0, tf.reduce_sum(input, 1, keep_dims=True)))
# Create model
# def nade_like(input, weights):
#     hidden1 = tf.matmul( tf.truediv(input, tf.maximum(1.0,tf.reduce_sum(input, 1, keep_dims=True)) ) , weights['hidden1'])
#
#     hidden2 = tf.sigmoid(tf.matmul(hidden1, weights['hidden2']))
#
#     # Output layer
#     out_layer = (tf.matmul(hidden2, tf.mul(weights['pad'],weights['out'])) + bias['out'])
#     return out_layer

def nade_like(input, target, weights, bias):

    hidden01 = tf.matmul(normalize(input), weights['M1']) # Vd

    hidden01 = tf.batch_matmul(tf.expand_dims(hidden01,2),tf.ones([batch_size,1,NUM_NOTES])) # Vd augmented to D across 2 dimension

    hidden02 = cumsum_weights(target, weights['M2'],D)

    hidden = hidden01 + hidden02

    #hidden02 = tf.matmul(norm_cumsum(target), weights['M2'])
    #hidden02 = (hidden02) + bias['M2']

    hidden = hidden01 + hidden02

    output = tf.reduce_sum(hidden,1)#tf.matmul(hidden, weights['W'])
    return output

def norm_cumsum(target):
    cum_sum = cumsum(target)
    return tf.truediv(cum_sum, tf.maximum(1.0,tf.reduce_sum(cum_sum, 1, keep_dims=True)))


def cumsum(target):
    triangle = ones_triangular(NUM_NOTES)#tf.constant(ones_triangular(NUM_NOTES))
    return tf.matmul(target,triangle)


def get_batch(data_set,id, stoch=False):
    if stoch:
        transpose_data_set = list(map(list, zip(*train_set)))
        batch = random.sample(transpose_data_set, batch_size)
        batch_input,batch_target = list(map(list, zip(*batch)))
        return batch_input,batch_target
    batch_id = id + 1
    input, target = data_set
    return input[(batch_id * batch_size - batch_size):(batch_id * batch_size)], target[(batch_id * batch_size - batch_size):(batch_id * batch_size)]

def cost_function():

    return tf.reduce_mean(tf.reduce_sum( tf.add(tf.log(pred)*target,
                             tf.log(tf.add(1.0,tf.neg(pred))) * tf.add(1.0, tf.neg(target))),1 ))
# Construct model


def train(checkpoint_path='save_models/test/nade_like_test.ckpt',load_model=None):

    print('Create model ...')

    pred = nade_like(input, target, weights,bias)
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(pred, target), 1))
    optimizer = tf.train.AdamOptimizer(epsilon=1e-03, learning_rate=learning_rate).minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    # Initializing the variables
    init = tf.initialize_all_variables()
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
    # Launch the graphx

    with tf.Session() as sess:
        if load_model:
            checkpoint = tf.train.get_checkpoint_state(load_model)
            if checkpoint and tf.gfile.Exists(checkpoint.model_checkpoint_path):
                print("Reading model parameters from %s" % checkpoint.model_checkpoint_path)
                saver.restore(sess, checkpoint.model_checkpoint_path)
            else:
                print("ooops no saved model found in %s ! " % load_model)
        else:
            print("using fresh parameters...")
            sess.run(init)

        print('Loading data ...')
        dic = pickle.load(open('JSB_processed.pkl', 'rb'))
        train_chords = dic['t']
        test_chords = dic['te']
        valid_chords = dic['v']

        train_set = dp.generate_binary_vectors(train_chords)
        input_train, target_train = train_set
        data_size = len(train_set[0])
        test_set = dp.generate_binary_vectors(test_chords)
        valid_set = dp.generate_binary_vectors(valid_chords)
        input_valid, target_valid = valid_set

        #input_valid, target_valid = test_set
        #input_train, target_train = train_set

        batch_vx, batch_vy = get_batch(valid_set, 0)
        best_val_loss = sess.run(cost, feed_dict={input: batch_vx, target: batch_vy})
        print('Start training ...')
        # Training cycle
        previous_eval_loss = []

        best_val_epoch = -1
        strikes = 0
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(data_size / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = get_batch(train_set, i)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c, out = sess.run([optimizer, cost, pred], feed_dict={input: batch_x,
                                                                         target: batch_y})

                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%d' % (epoch + 1), "cost=", \
                      "{:.9f}".format(avg_cost))

            batch_vx, batch_vy = get_batch(valid_set, 1)
            c_valid = sess.run(cost, feed_dict={input: batch_vx, target: batch_vy})
            print("Valid error %4f" % (c_valid))
            previous_eval_loss.append(c_valid)

            improve_valid = previous_eval_loss[-1] < best_val_loss

            if improve_valid:
                best_val_loss = previous_eval_loss[-1]
                best_val_epoch = epoch
                # Save checkpoint.
                saver.save(sess, checkpoint_path, global_step=epoch)
            else:
                strikes += 1
            if strikes > 3:
                break
        print("Optimization Finished!")

        input_test, target_test = test_set
        batch_tex, batch_tey = get_batch(test_set, 1)
        c_test = sess.run(cost, feed_dict={input: batch_tex, target: batch_tey})

        batch_x, batch_y = get_batch(train_set, 0)
        c_train = sess.run(cost, feed_dict={input: batch_x, target: batch_y})

        print("Train error %.9f" % (c_train))
        print("Best validation %.9f" % (best_val_loss))
        print("Test error %.9f" % (c_test))

def print_error(sess, weights=None, bias=None,checkpoint_path="save_models/new", num_notes=None, d=None):

    D = 1024
    NUM_NOTES = 88
    if num_notes:
        NUM_NOTES=num_notes
    if d:
        D = d
    # tf Graph input
    input = tf.placeholder("float", [None, NUM_NOTES])
    target = tf.placeholder("float", [None, NUM_NOTES])

    print('Loading data ...')
    dic = pickle.load(open('JSB_processed.pkl', 'rb'))
    train_chords = dic['t']
    test_chords = dic['te']
    valid_chords = dic['v']

    train_set = dp.generate_binary_vectors(train_chords)
    input_train, target_train = train_set
    test_set = dp.generate_binary_vectors(test_chords)
    input_test, target_test = test_set
    valid_set = dp.generate_binary_vectors(valid_chords)
    input_valid, target_valid = valid_set

    # Store layers weight & bias
    if weights is None:
        weights = {
            'M1': tf.Variable(tf.random_normal([NUM_NOTES, D])),
            'M2': tf.Variable(tf.random_normal([NUM_NOTES, D])),
            # 'hidden2': tf.Variable(tf.random_normal([D, NUM_NOTES])),
            # 'pad' : tf.Variable(tf.constant(ones_triangular(NUM_NOTES).tolist()),trainable=False),
            'W': tf.Variable(tf.random_normal([D, NUM_NOTES]))
        }
    if bias is None:
        bias = {
            #     'hidden1': tf.Variable(tf.random_normal([D])),
            'M2': tf.Variable(tf.random_normal([D])),
            #     'out': tf.Variable(tf.random_normal([NUM_NOTES]))
        }
    pred = nade_like(input, target, weights, bias)
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(pred, target), 1))
    optimizer = tf.train.AdamOptimizer(epsilon=1e-03, learning_rate=learning_rate).minimize(cost)
    #with tf.Session() as sess:
    checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
    if checkpoint and tf.gfile.Exists(checkpoint.model_checkpoint_path):
        print("Reading model parameters from %s" % checkpoint.model_checkpoint_path)
        saver.restore(sess, checkpoint.model_checkpoint_path)
    else:
        print("ooops no saved model found in %s ! " % checkpoint_path)

    c = sess.run(cost, feed_dict={input: input_train, target: target_train})
    print("train cost")
    print(c)
    c = sess.run(cost, feed_dict={input: input_valid, target: target_valid})
    print("valid cost")
    print(c)
    c = sess.run(cost, feed_dict={input: input_test, target: target_test})
    print("test cost")
    print(c)

if __name__ == "__main__":
    train()