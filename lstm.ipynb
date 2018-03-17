# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import tensorflow as tf
import numpy as np
import collections
import os
import argparse
import datetime as dt
import re 
import pandas as pd 
sess = tf.InteractiveSession()
#from tensorflow.python import debug as tf_debug


"""To run this code, you'll need to first download and extract the text dataset
    from here: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz. Change the
    data_path variable below to your local exraction path"""


# Look at the data
data_path = "/Users/jbo50/Desktop"
fileloc= '/Users/jbo50/Downloads/CustomerActions-Movies-TimedUUID-3days.csv'
# data = pd.read_csv('/Users/jbo50/Downloads/CustomerActions-Movies-TimedUUID-3days.csv')
data_dict={}
max_len=77
with open(fileloc,'r') as f:
    lis=[re.split(",|~|\n",line) for line in f]
for i,v in enumerate(lis):   
    lis2 = list(filter(None, v))
    padded_list = lis2 + [0]* (max_len-len(lis2))
    data_dict[i]= padded_list #remove empty entries
#     if len(v) > max_len:
#         max_len = len(v)

df= pd.DataFrame(data_dict).T

only_uuids = df.loc[:,::2] #select only uuid not timestamps 
#only_uuids.head()
  
set_uuid = set([])
for i in range(2,max_len,2): #start from 2 because 1 is the device id 
    set_uuid.update(only_uuids.loc[:,i])
set_uuid.remove(0)

vocabulary = len(set_uuid) #length of vocabulary
dictionary={}  # uuid:int
reversed_dictionary={}  #int:uuid
# convert uuid to numbers 
for i,value in enumerate(set_uuid):
    reversed_dictionary[i] = value
    dictionary[value]=i
    
#Lets look at user with at least for watched items
four_or_more = only_uuids.loc[np.sum(only_uuids!=0,axis=1)>4,:]
#four_or_more.head()

total_samples = len(four_or_more.index)
input_=four_or_more.loc[:,[2,4,6]].as_matrix() #use the first 3 uuids as input 
target_ = four_or_more.loc[:,[4,6,8]].as_matrix()   #use the fourth uuid as target 


# integer encode input data
input_encoded_all=[]
for i in range(len(input_)):
    input_encoded_all.append([dictionary[uuid] for uuid in input_[i]])
    
input_encoded_all= [item for s in input_encoded_all for item in s]

#put convert uuid to number and put them as a long list 
target_all=[]
for i in range(len(target_)):
    target_all.append([dictionary[uuid] for uuid in target_[i]])
target_all= [item for s in target_all for item in s]


#onehot_target_all = list()
#for value in target_all:
#    temp = [0] * len(set_uuid)
#    temp[value] = 1
#    onehot_target_all.append(temp)

#onehot_target_all=[item for s in onehot_target_all for item in s]
#Reshape training to [no_samples,timestep,number of feature]


# train/test split 80/20
np.random.seed(1)
train_idx =np.random.choice(list(range(len(onehot_target_all))),size=round(0.8*len(onehot_target_all)),replace=False)
temp_idx = set(list(range(len(onehot_target_all))))-set(train_idx)
np.random.seed(1)
test_idx =np.random.choice(list(temp_idx),size=round(0.5*len(temp_idx)),replace=False)
valid_idx = set(temp_idx)-set(test_idx)

train_input= [input_encoded_all[i] for i in train_idx]
train_target = [onehot_target_all[i] for i in train_idx]

test_input= [input_encoded_all[i] for i in test_idx]
test_target = [onehot_target_all[i] for i in test_idx]

valid_input= [input_encoded_all[i] for i in test_idx]
valid_target = [onehot_target_all[i] for i in test_idx]


#train_data = train_input[:1000]
#train_target =train_target[:1000]
train_data = input_encoded_all
train_target = target_all

train_data = train_data[:60000]
train_target = train_target[:60000]

test_data = train_data[60000:]
test_target = train_target[60000:]






print(len(train_input),len(train_target))
print(len(test_input),len(test_target))
print(len(valid_input),len(valid_target))
#

def batch_producer(raw_data, raw_target,batch_size, num_steps):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
    raw_target = tf.convert_to_tensor(raw_target, name="raw_target", dtype=tf.int32)
    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0: batch_size * batch_len],
                      [batch_size, batch_len])
    targ = tf.reshape(raw_target[0: batch_size * batch_len],
                      [batch_size, batch_len])
    epoch_size = (batch_len - 1) // num_steps
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    y = targ[:, i * num_steps:(i + 1) * num_steps]
    y.set_shape([batch_size, num_steps])
    x = data[:, i * num_steps:(i + 1) * num_steps]
    x.set_shape([batch_size, num_steps])
    return x,y


class Input(object):
    def __init__(self, batch_size, num_steps, data,target):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
#        self.input_data = data
#        self.targets = target
        self.input_data,self.targets = batch_producer(raw_data=data, raw_target=target,batch_size=batch_size, num_steps=num_steps) 
        
#        self.targets = batch_producer(target, batch_size, num_steps)


# create the main model
class Model(object):
    def __init__(self, input, is_training, hidden_size, vocab_size, num_layers,
                 dropout=0.5, init_scale=0.05):
        self.is_training = is_training
        self.input_obj = input
        self.batch_size = input.batch_size
        self.num_steps = input.num_steps
        self.hidden_size = hidden_size

        # create the word embeddings
        with tf.device("/cpu:0"):
            embedding = tf.Variable(tf.random_uniform([vocab_size, self.hidden_size], -init_scale, init_scale))
            inputs = tf.nn.embedding_lookup(embedding, self.input_obj.input_data)
#            print('target shape',self.input_obj.targets.shape)
        if is_training and dropout < 1:
            inputs = tf.nn.dropout(inputs, dropout)

        # set up the state storage / extraction
        self.init_state = tf.placeholder(tf.float32, [num_layers, 2, self.batch_size, self.hidden_size])

        state_per_layer_list = tf.unstack(self.init_state, axis=0)
        rnn_tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
             for idx in range(num_layers)]
        )

        # create an LSTM cell to be unrolled
        cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0)
        # add a dropout wrapper if training
        if is_training and dropout < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple=True)

        output, self.state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, initial_state=rnn_tuple_state)
        # reshape to (batch_size * num_steps, hidden_size)
        output = tf.reshape(output, [-1, hidden_size])

        softmax_w = tf.Variable(tf.random_uniform([hidden_size, vocab_size], -init_scale, init_scale))
        softmax_b = tf.Variable(tf.random_uniform([vocab_size], -init_scale, init_scale))
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])
        # Use the contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            self.input_obj.targets,
            tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)
        # Update the cost
        self.cost = tf.reduce_sum(loss)

        # get the prediction accuracy
        self.softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, vocab_size]))
        self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int32)
        correct_prediction = tf.equal(self.predict, tf.reshape(self.input_obj.targets, [-1]))
#        print('pre',self.predict.shape)
#        print('act',self.input_obj.targets.shape)
#        print('cr',correct_prediction.shape)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.histogram('softmax', self.softmax_out)
        tf.summary.histogram('prediction', self.predict)
        tf.summary.histogram('cross_entropy', loss)
        tf.summary.histogram('accuracy', self.accuracy)


        if not is_training:
           return
        self.learning_rate = tf.Variable(0.0, trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
#        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step()) 
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

        self.new_lr = tf.placeholder(tf.float32, shape=[])
        self.lr_update = tf.assign(self.learning_rate, self.new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})

data_path ='/Users/jbo50/Documents/researching/lstm_models'
def train(train_data, vocabulary, num_layers, num_epochs, batch_size, model_save_name,
          learning_rate=0.005, max_lr_epoch=10, lr_decay=0.93, print_iter=50):
    tf.reset_default_graph()
    # setup data and models
    training_input = Input(batch_size=batch_size, num_steps=3, data=train_data,target=train_target)
    m = Model(training_input, is_training=True, hidden_size=600, vocab_size=vocabulary,
              num_layers=num_layers)
    init_op = tf.global_variables_initializer()
    orig_decay = lr_decay
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter( '/Users/jbo50/Documents/researching/logs/1/train', sess.graph)
        merge = tf.summary.merge_all()
        # start threads
        sess.run([init_op])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver = tf.train.Saver()
        for epoch in range(num_epochs):
            new_lr_decay = orig_decay ** max(epoch + 1 - max_lr_epoch, 0.0)
            m.assign_lr(sess, learning_rate * new_lr_decay)
            current_state = np.zeros((num_layers, 2, batch_size, m.hidden_size))
            curr_time = dt.datetime.now()
            for step in range(training_input.epoch_size):
                # cost, _ = sess.run([m.cost, m.optimizer])
                if step % print_iter != 0:
                    summary,cost, _, current_state = sess.run([merge,m.cost, m.train_op, m.state],
                                                      feed_dict={m.init_state: current_state})
                    train_writer.add_summary(summary, epoch)
                else:
                    seconds = (float((dt.datetime.now() - curr_time).seconds) / print_iter)
                    curr_time = dt.datetime.now()
                    summary,cost, _, current_state, acc = sess.run([merge,m.cost, m.train_op, m.state, m.accuracy],
                                                           feed_dict={m.init_state: current_state})
                    train_writer.add_summary(summary, epoch)
                    print("Epoch {}, Step {}, cost: {:.3f}, accuracy: {:.3f}, Seconds per step: {:.3f}".format(epoch,
                            step, cost, acc, seconds))
            
            # save a model checkpoint
            saver.save(sess, data_path + '/' + model_save_name, global_step=1000)
        # do a final save
        saver.save(sess, data_path + '/' + model_save_name + '-final')
        # close threads
        coord.request_stop()
        coord.join(threads)


def test(model_path, test_data, reversed_dictionary):
    test_input = Input(batch_size=50, num_steps=3, data=train_data,target=train_target)
    m = Model(test_input, is_training=False, hidden_size=600, vocab_size=vocabulary,
              num_layers=3)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # start threads
        
#        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        current_state = np.zeros((2, 2, m.batch_size, m.hidden_size))
        # restore the trained model
        saver.restore(sess, model_path)
        # get an average accuracy over num_acc_batches
        num_acc_batches = 30
        check_batch_idx = 25
        acc_check_thresh = 5
        accuracy = 0
        for batch in range(num_acc_batches):
            if batch == check_batch_idx:
                true_vals, pred, current_state, acc = sess.run([m.input_obj.targets, m.predict, m.state, m.accuracy],
                                                               feed_dict={m.init_state: current_state})
                pred_string = [reversed_dictionary[x] for x in pred[:m.num_steps]]
                true_vals_string = [reversed_dictionary[x] for x in true_vals[0]]
                print("True values (1st line) vs predicted values (2nd line):")
                print(" ".join(true_vals_string))
                print(" ".join(pred_string))
            else:
                acc, current_state = sess.run([m.accuracy, m.state], feed_dict={m.init_state: current_state})
            if batch >= acc_check_thresh:
                accuracy += acc
        print("Average accuracy: {:.3f}".format(accuracy / (num_acc_batches-acc_check_thresh)))
        # close threads
        coord.request_stop()
        coord.join(threads)




train(train_data, vocabulary, num_layers=3, num_epochs=100, batch_size=50,
          model_save_name='three-layer-lstm-medium-config-60-epoch-0p93-lr-decay-10-max-lr')

trained_model = data_path + "/three-layer-lstm-medium-config-60-epoch-0p93-lr-decay-10-max-lr-final"
test(trained_model, train_data, reversed_dictionary)