import sys
sys.path.append('../sentiment_analysis/')

import argparse
import tensorflow as tf
import numpy as np
import tensorflow.contrib.seq2seq as seq2seq
from util import *
import os

class model():
    def __init__(self, args):
        self.args = args 
         
        self.pos_inps = tf.placeholder(dtype=tf.int32, shape=[None, args.max_time_step, 1])
        self.neg_inps = tf.placeholder(dtype=tf.int32, shape=[None, args.max_time_step, 1])

        converted_neg = self.converter(self.pos_inps, "converter_pos2neg")
        converted_pos = self.converter(self.neg_inps, "converter_neg2pos")

        dis_pos = self.discriminator(self.pos_inps, "dis_pos")
        dis_fake_pos = self.discriminator(converted_pos, "dis_pos", reuse=True)

        dis_neg = self.discriminator(self.neg_inps, "dis_neg")
        dis_fake_neg = self.discriminator(converted_neg, "dis_neg", reuse=True)

        self. loss_d_p = tf.nn.softmax_cross_entropy_with_logits(logits=dis_pos, labels=tf.ones_like(dis_pos)) + tf.nn.softmax_cross_entropy_with_logits(logits=dis_fake_pos, labels=tf.zeros_like(dis_fake_pos))
        self.loss_d_n = tf.nn.softmax_cross_entropy_with_logits(logits=dis_neg, labels=tf.ones_like(dis_neg)) + tf.nn.softmax_cross_entropy_with_logits(logits=dis_fake_neg, labels=tf.zeros_like(dis_fake_neg))

        self.loss_g_p = tf.nn.softmax_cross_entropy_with_logits(logits=dis_fake_pos, labels=tf.ones_like(dis_fake_pos))
        self.loss_g_n = tf.nn.softmax_cross_entropy_with_logits(logits=dis_fake_neg, labels=tf.ones_like(dis_fake_neg))

        var_ = tf.global_variables()
        var_d_p = [var for var in var_ if var.name == "dis_pos"]
        var_d_n = [var for var in var_ if var.name == "dis_neg"]
        var_g_p = [var for var in var_ if var.name == "converter_neg2pos"]
        var_g_n = [var for var in var_ if var.name == "converter_pos2neg"]

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
           
            with tf.variable_scope("CNN") as scope:
                convded = []
                for kernel, filter_num in zip(self.args.kernels, self.args.filter_nums):
                    conv_ = tf.layers.conv2d(cnn_inputs, filter_num, kernel_size=[kernel, self.args.embedding_size], strides=[1, 1], activation=tf.nn.relu, padding='valid', name="conv_{}".format(kernel))
                    pool_ = tf.layers.max_pooling2d(conv_, pool_size=[self.args.max_time_step-kernel+1, 1], padding='valid', strides=[1, 1])
                    convded.append(tf.reshape(pool_, (-1, filter_num)))
                convded = tf.concat([cnn_output for cnn_output in convded], axis=-1)
        
            with tf.variable_scope("Dense") as scope:
                flatten_ = tf.contrib.layers.flatten(convded)
                logits = tf.layers.dense(flatten_, 1, name="dense_layer")
           
            return logits

    def train(self):
        opt_d_p = tf.train.AdamOptimizer(self.args.lr).minimize(self.loss_d_p)
        opt_d_n = tf.train.AdamOptimizer(self.args.lr).minimize(self.loss_d_n)
        opt_g_p = tf.train.AdamOptimizer(self.args.lr).minimize(self.loss_g_p)
        opt_g_n = tf.train.AdamOptimizer(self.args.lr).minimize(self.loss_g_n)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            graph = tf.summary.FileWriter('./logs', sess.graph)

            for itr in range(self.args.itrs):

                if itr % 20 != 0:
                    pass

                if itr % 1000 != 0:
                    saver.save(sess, "saved/model.ckpt")
                    print("----------------------saved model-------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--lr", dest="lr", type=float, default= 0.2)
    parser.add_argument("--data_dir", dest="data_dir", default="../data/")
    parser.add_argument("--index_dir", dest="index_dir", default="../data/index.txt")
    parser.add_argument("--itrs", dest="itrs", type=int, default=10001)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=40)
    parser.add_argument("--embedding_size", dest="embedding_size", default=64)
    parser.add_argument("--max_time_step", dest="max_time_step", type=int, default=20)
    parser.add_argument("--vocab_size", dest="vocab_size", type=int, default=2348)
    parser.add_argument("--train", dest="train", type=bool, default=True)
    parser.add_argument("--kernels", dest="kernels" type=list, default=[2,3,4,5,6])
    parser.add_argument("--filter_nums", dest="filter_nums", type=list, default=[32,64,128,128,224])
    parser.add_argument("--test", dest="test", type=bool, default=True)
    args= parser.parse_args()
    
    if not os.path.exists("save"):
        os.mkdir("save")

    if not os.path.exists("logs"):
        os.mkdir("logs")

    model_ = model(args)
    if args.train:
        model_.train()

