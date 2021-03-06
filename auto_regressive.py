'''
A tensorflow implementation of the Autogressive chord2vec model
'''

import tensorflow as tf
from chord2vec.linear_models import data_processing as dp
import pickle
import numpy as np
import random
import sys
from packaging import version

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_math_ops

import logging
logger = logging.getLogger(__name__)

def about(x, LINE=80, SINGLE_LINE=False):
    '''
    author: Yogesh Garg (https://github.com/yogeshg)
    '''
    s ='type:'+str(type(x))+' '
    try:
        s+='shape:'+str(x.shape)+' '
    except Exception as e:
        pass
    try:
        s+='dtype:'+str(x.dtype)+' '
    except Exception as e:
        pass
    try:
        s1 = str(x)
        if(SINGLE_LINE):
            s1 = ' '.join(s1.split('\n'))
            extra = (len(s)+len(s1)) - LINE
            if(extra > 0):
                s1 = s1[:-(extra+3)]+'...'
            s+=s1
        else:
            s+='\n'+s1
    except Exception as e:
        pass
    return s


# Parameters
learning_rate = 0.002
training_epochs = 20
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
    'W': tf.Variable(tf.random_normal([D, NUM_NOTES]))
}

bias = {
    'M2': tf.Variable(tf.random_normal([D])),
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
    return tf.matmul(tf.ones([batch_size, r, 1]), tf.expand_dims(input, 1))

def mask(input, W, r=D):
    inputs = extend_vector(input,r,batch_size)
    if oldversion:
        return  tf.squeeze(tf.mul(inputs, [W]))
    else:
        return  tf.squeeze(tf.multiply(inputs, [W]))

def cumsum_weights(input, W, r=D):
    masked=mask(input,W,r)
    triangle = ones_triangular(NUM_NOTES)
    size = batch_size
    if oldversion:
        return tf.batch_matmul(masked, np.array([triangle]*size))
    else:
        return tf.matmul(masked, np.array([triangle]*size))


def normalize(input):
    return tf.truediv(input, tf.maximum(1.0, tf.reduce_sum(input, 1, keep_dims=True)))


def auto_regressive_model(input, target, weights, bias):
    """
    Builds the auto regressive model. For details on the model, refer to the written report
    """
    if oldversion:
        hidden01 = tf.batch_matmul(normalize(input), weights['M1']) # V_d

        hidden01 = tf.batch_matmul(tf.expand_dims(hidden01,2),tf.ones([batch_size,1,NUM_NOTES])) # V_d augmented to D across  dimension 2
    else:
        hidden01 = tf.matmul(normalize(input), weights['M1']) # V_d

        hidden01 = tf.matmul(tf.expand_dims(hidden01,2),tf.ones([batch_size,1,NUM_NOTES])) # V_d augmented to D across  dimension 2

    hidden02 = cumsum_weights(normalize(target), weights['M2'],D)  # V_c

    hidden = hidden01 + hidden02

    y = tf.zeros([1], tf.float32)
    if oldversion:
        split = tf.split(0, batch_size, hidden)
    else:
        split = tf.split(hidden, batch_size, 0)
    if oldversion:
        y = tf.batch_matmul(tf.expand_dims(tf.transpose(tf.squeeze(split[0])), 1), tf.expand_dims(tf.transpose(weights['W']), 2))
    else:
        y = tf.matmul(
                    tf.expand_dims(tf.transpose(tf.squeeze(split[0])), 1),
                    tf.expand_dims(tf.transpose(weights['W']), 2)
                    )

    s0 = about(y)
    for i in range(1, len(split)):
        if oldversion:
            y = tf.concat(0, [y, tf.batch_matmul(tf.expand_dims(tf.transpose(tf.squeeze(split[i])), 1),
                                                     tf.expand_dims(tf.transpose(weights['W']), 2))])
        else:
            y = tf.concat( [y, tf.matmul(
                                tf.expand_dims(tf.transpose(tf.squeeze(split[i])), 1),
                                tf.expand_dims(tf.transpose(weights['W']), 2)
                            )],0)
    s1 = about(y)
    y = tf.squeeze(y)
    s2 = about(y)
    output = tf.reshape(y,[batch_size,NUM_NOTES])

    logger.info('hidden01:\n'+ about(hidden01))
    logger.info('hidden02:\n'+ about(hidden02))
    logger.info('hidden:\n'+ about(hidden))
    logger.info('y0:\n'+ s0)
    logger.info('y1:\n'+ s1)
    logger.info('y2:\n'+ s2)
    logger.info('output:\n'+ about(output))

    return output

def norm_cumsum(target):
    """ Normalized cumulative sum"""
    cum_sum = cumsum(target)
    return tf.truediv(cum_sum, tf.maximum(1.0,tf.reduce_sum(cum_sum, 1, keep_dims=True)))


def cumsum(target):
    """ Cumulative sum"""
    triangle = ones_triangular(NUM_NOTES)#tf.constant(ones_triangular(NUM_NOTES))
    return tf.matmul(target,triangle)


def get_batch(data_set,id, stoch=False):
    if stoch:
        transpose_data_set = list(map(list, zip(*data_set)))
        batch = random.sample(transpose_data_set, batch_size)
        batch_input,batch_target = list(map(list, zip(*batch)))
        return batch_input,batch_target
    batch_id = id + 1
    input, target = data_set
    return input[(batch_id * batch_size - batch_size):(batch_id * batch_size)], target[(batch_id * batch_size - batch_size):(batch_id * batch_size)]

# Construct model

def load_data(file_name = "JSB_Chorales.pickle"):
    print('Loading data ...')
    train_chords, test_chords , valid_chords = dp.read_data(file_name,1)

    train_set = dp.generate_binary_vectors(train_chords)
    # input_train, target_train = train_set
    test_set = dp.generate_binary_vectors(test_chords)
    valid_set = dp.generate_binary_vectors(valid_chords)
    # input_valid, target_valid = valid_set

    data_size = len(train_set[0])
    data_size_valid = len(valid_set[0])
    data_size_te = len(test_set[0])

    total_batch = int(data_size / batch_size)
    total_batch_valid = int(data_size_valid / batch_size)
    total_batch_test = int(data_size_te / batch_size)

    return train_set, test_set, valid_set, total_batch, total_batch_test, total_batch_valid

def train(file_name,checkpoint_path='save_models/nade3/nade_like_D1024_batch128.ckpt', load_model=None, print_train=False):
    print('Create model ...')

    pred = auto_regressive_model(input, target, weights, bias)
    # Define loss and optimizer

    cost = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=target), 1))

    optimizer = tf.train.AdamOptimizer(epsilon=1e-00, learning_rate=learning_rate).minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    if oldversion:
        init = tf.initialize_all_variables()
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
    else:
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    # Launch the graphx

    if True: #with tf.Session() as sess:
        if load_model:
             checkpoint = tf.train.get_checkpoint_state(load_model)
             print(checkpoint.model_checkpoint_path)
             print(tf.gfile.Exists(checkpoint.model_checkpoint_path))
             if checkpoint and tf.gfile.Exists(checkpoint.model_checkpoint_path):
                 print("Reading model parameters from %s" % checkpoint.model_checkpoint_path)
                 saver.restore(sess, checkpoint.model_checkpoint_path)
             else:
                 print("ooops no saved model found in %s ! " % load_model)
                 sys.exit()
            #saver.restore(sess, load_model)
        else:
            print("using fresh parameters...")
            sess.run(init)

        train_set, test_set, valid_set, total_batch, total_batch_test, total_batch_valid = load_data(file_name)

        batch_vx, batch_vy = get_batch(valid_set, 0)
        best_val_loss = sess.run(cost, feed_dict={input: batch_vx, target: batch_vy})
        print('Start training ...')


        # Training cycle
        previous_eval_loss = []

        best_val_epoch = -1
        strikes = 0
        for epoch in range(training_epochs):
            print('epoch:', epoch, 'of', training_epochs,"\n")
            avg_cost = 0.

            # Loop over all batches
            for i in range(total_batch):
                if i % 50==0:
                    print('batch:', i, 'of', total_batch)
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

            avg_cost_valid = 0.
            for batch_id in range(total_batch_valid):
                batch_vx, batch_vy = get_batch(valid_set, batch_id)
                c_valid = sess.run(cost, feed_dict={input: batch_vx, target: batch_vy})
                avg_cost_valid += c_valid / total_batch_valid

            print("Valid error %4f" % (avg_cost_valid))
            previous_eval_loss.append(avg_cost_valid)

            improve_valid = previous_eval_loss[-1] < best_val_loss

            if improve_valid:
                best_val_loss = previous_eval_loss[-1]
                best_val_epoch = epoch
                # Save checkpoint.
                saver.save(sess, checkpoint_path, global_step=epoch)
            else:
                strikes += 1
            if strikes > 5:
                break
        print("Optimization Finished!")
        print("Best validation at epoch: %d" %best_val_epoch)

        input_test, target_test = test_set
        avg_cost_test = 0.
        for batch_id in range(total_batch_test):
            batch_tex, batch_tey = get_batch(test_set, 0)
            c_test = sess.run(cost, feed_dict={input: batch_tex, target: batch_tey})
            avg_cost_test += c_test / total_batch_test

        #batch_x, batch_y = get_batch(train_set, 0)
        #c_train = sess.run(cost, feed_dict={input: batch_x, target: batch_y})
        if print_train:
            avg_cost = 0.
            for batch_id in range(total_batch):
                batch_x, batch_y = get_batch(train_set, batch_id)
                c = sess.run(cost, feed_dict={input: batch_x, target: batch_y})
                avg_cost += c / total_batch
            print("train cost")
            print(c)

        #print("Train error %.9f" % (c_train))
        print("Best validation %.9f" % (best_val_loss))
        print("Test error %.9f" % (avg_cost_test))

def print_error(file_name,checkpoint_path="save_models/new", print_train=False, print_valid=False):
    print('Create model ...')

    pred = auto_regressive_model(input, target, weights, bias)
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(pred, target), 1))
    optimizer = tf.train.AdamOptimizer(epsilon=1e-03, learning_rate=learning_rate).minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    # Initializing the variables
    init = tf.initialize_all_variables()
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
    # Launch the graphx

    with tf.Session() as sess:
        if checkpoint_path:
            saver.restore(sess, checkpoint_path)
        else:
            print("using fresh parameters...")
            sess.run(init)

        train_set, test_set, valid_set, total_batch, total_batch_test, total_batch_valid = load_data(file_name)

        if print_train:
            avg_cost = 0.
            for batch_id in range(total_batch):
                batch_x, batch_y = get_batch(train_set, batch_id)
                c = sess.run(cost, feed_dict={input: batch_x, target: batch_y})
                avg_cost += c / total_batch
            print("train cost")
            print(c)

        if print_valid:
            avg_cost_valid = 0.
            for batch_id in range(total_batch_valid):
                batch_vx, batch_vy = get_batch(valid_set, batch_id)
                c_valid = sess.run(cost, feed_dict={input: batch_vx, target: batch_vy})
                avg_cost_valid += c_valid / total_batch_valid
            print("valid cost")
            print(avg_cost_valid)

        avg_cost_test = 0.
        for batch_id in range(total_batch_test):
            batch_tex, batch_tey = get_batch(test_set, batch_id)
            c_test = sess.run(cost, feed_dict={input: batch_tex, target: batch_tey})
            avg_cost_test += c_test / total_batch_test
        print("test cost")
        print(avg_cost_test)

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', required=True, type=str, help='location of pickle file to train on')
    parser.add_argument('--checkpoint_path', required=True, type=str, help='path to save model checkpoints')
    parser.add_argument('--load_model', default=None, help='pre trained model to resume training')
    parser.add_argument('--print_train', action='store_true', default=False, help='verbosity')
    args = parser.parse_args(sys.argv[1:])
    oldversion=True
    sess = tf.InteractiveSession()
    if version.parse(tf.__version__) > version.parse("0.11.0"):
         oldversion=False
    train(**args.__dict__)
    # train("JSB_Chorales.pickle", checkpoint_path="save_models/new", load_model="save_models")

