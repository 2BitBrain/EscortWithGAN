import sys
sys.path.append('../sentiment_analysis/')


import tensorflow as tf
import numpy as np
import tensorflow.contrib.seq2seq as seq2seq
from util import *

class model():
    def __init__(self, args):
        self.args = args 
         
        self.pos_inps = tf.placeholder(dtype=tf.int32, shape=[None, args.max_time_step, 1])
        self.pos_labels = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        
        self.neg_inps = tf.placeholder(dtype=tf.int32, shape=[None, args.max_time_step, 1])
        self.neg_labels = tf.placeholder(dtype=tf.float32, shape=[None, 2])

    def def_cell(self):
        if self.args.cell_model == 'rnn':
            cell_fn = tf.contrib.rnn.BasicRNNCell
        elif self.args.cell_model == 'gru':
            cell_fn = tf.contrib.rnn.GRUCell
        elif self.args.cell_model == 'lstm':
            cell_fn = tf.contrib.rnn.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(self.args.cell_model))

        def cell():
            cell_ = cell_fn(self.args.rnn_size, reuse=tf.get_variable_scope().reuse)
            if self.args.keep_prob < 1.:
                cell_ = tf.contrib.rnn.DropoutWrapper(cell_, output_keep_prob=self.args.keep_prob)
            return cell_
        
        cell = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(self.args.num_layers)], state_is_tuple = True)
        return cell
            
    def converter(self, x, name, reuse=False):
        ##using word level cnn's feature
        ##input shapes are (None, max_timestep, 1)
        ##output shape are (None, max_timestep, 1) and return index which is highest probablistic.
        with tf.variable_scope(name) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope('Embedding') as scope:
                rnn_inputs = []
                embedding_weight = tf.Variable(tf.random_uniform([self.args.vocab_size, self.args.embedding_size],-1.,1.), name='embedding_weight')
                word_t  = tf.split(self.inputs, self.args.max_time_step, axis=1)
                
                for t in range(self.args.max_time_step):
                    char_index = tf.reshape(word_t[t], shape=[-1, self.args.max_word_length])
                    embedded = tf.nn.embedding_lookup(embedding_weight, char_index)
                    rnn_inputs.append(embedded)
                rnn_inputs = tf.convert_to_tensor(rnn_inputs)

            with tf.variable_scope('Encoder') as scope:
                encoder_cell = tf.contrib.rnn.MultiRNNCell(self.def_cell(), state_is_tuple = True)     
                _, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, self.encoder_input, initial_state=encoder_cell.zero_state(batch_size=self.args.batch_size, dtype=tf.float32), dtype=tf.float32)

            with tf.variable_scope("Decoder") as scope:
                decoder_cell = tf.contrib.rnn.MultiRNNCell(self.def_cell(), state_is_tuple = True)
                helper = seq2seq.TrainingHelper(self.decoder_input, tf.cast([self.args.max_time_step]*self.args.batch_size, tf.int32))
                decoder = seq2seq.BasicDecoder(decoder_cell, helper=helper, initial_state=encoder_final_state)
                decoder_outputs_, decoder_final_state, _ =  seq2seq.dynamic_decode(decoder= decoder)
                decoder_outputs, sample_id = decoder_outputs_

            with tf.variable_scope("Outputs") as scope:
                logits = []
                for t in self.args.max_time_step:
                    if t != 0:
                        tf.get_variable_scope().reuse_variables()

                    logit = tf.layers.dense(decoder_outputs[t], self.args.vocab_size, name="dense")
            
                    logits.append(logit)
                logits = tf.convert_to_tensor(logits)
            
            return tf.reduce_max(logits, axis=-1)

    def discriminator(self, x, name, reuse=False): 
        with tf.variable_scope(name) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
              
            with tf.variable_scope("Embedding") as scope:
                splitted_word_ids  = tf.split(x, self.args.max_time_step, axis=1)
                embedding_weight = tf.Variable(tf.random_uniform([args.vocab_size, args.embedding_size],-1.,1.), name='embedding_weight')
                t_embedded = []
            
                for t in range(self.args.max_time_step):
                    if t is not 0:
                        tf.get_variable_scope().reuse_variables()

                    embedded = tf.nn.embedding_lookup(embedding_weight, self.inputs[:,t,:])
                    t_embedded.append(embedded)
                cnn_inputs = tf.reshape(tf.transpose(tf.convert_to_tensor(t_embedded), perm=(1,0,2,3)), (-1, self.args.max_time_step, self.args.embedding_size,1))
           
            kernels = [2,3,4,5,6]
            filter_nums = [32,64,128,128,224]
            with tf.variable_scope("CNN") as scope:
                convded = []
                for kernel, filter_num in zip(kernels, filter_nums):
                    conv_ = tf.layers.conv2d(cnn_inputs, filter_num, kernel_size=[kernel, self.args.embedding_size], strides=[1, 1], activation=tf.nn.relu, padding='valid', name="conv_{}".format(kernel))
                    pool_ = tf.layers.max_pooling2d(conv_, pool_size=[self.args.max_time_step-kernel+1, 1], padding='valid', strides=[1, 1])
                    convded.append(tf.reshape(pool_, (-1, filter_num)))
                convded = tf.concat([cnn_output for cnn_output in convded], axis=-1)
        
            with tf.variable_scope("Dense") as scope:
                flatten_ = tf.contrib.layers.flatten(convded)
                logits = tf.layers.dense(flatten_, 2, name="dense_layer")
           
            return logits
