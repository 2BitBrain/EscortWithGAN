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
        
        converted_neg_pos = self.converter(converted_neg, "converter_neg2pos", reuse=True)
        converted_pos_neg = self.converter(converted_pos, "converter_pos2neg", reuse=True)

        dis_pos = self.discriminator(self.pos_inps, "dis_pos")
        dis_fake_pos = self.discriminator(converted_pos, "dis_pos", reuse=True)

        dis_neg = self.discriminator(self.neg_inps, "dis_neg")
        dis_fake_neg = self.discriminator(converted_neg, "dis_neg", reuse=True)

        self.loss_d_p = -1*(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dis_pos, labels=tf.ones_like(dis_pos))) + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dis_fake_pos, labels=tf.zeros_like(dis_fake_pos))))
        self.loss_d_n = -1*(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dis_neg, labels=tf.ones_like(dis_neg))) + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dis_fake_neg, labels=tf.zeros_like(dis_fake_neg))))
        
        cycle_loss = tf.to_float(tf.reduce_mean(tf.abs(tf.subtract(converted_neg_pos, self.pos_inps)))) + tf.to_float(tf.reduce_mean(tf.abs(tf.subtract(converted_pos_neg, self.neg_inps)))) 

        self.loss_g_p = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dis_fake_pos, labels=tf.ones_like(dis_fake_pos))) + tf.reduce_mean(args.l1_lambda*cycle_loss)
        self.loss_g_n = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dis_fake_neg, labels=tf.ones_like(dis_fake_neg))) + tf.reduce_mean(args.l1_lambda*cycle_loss)

        with tf.variable_scope("summary") as scope:
            tf.summary.scalar("discriminator_pos_loss", self.loss_d_p)
            tf.summary.scalar("discriminator_neg_loss", self.loss_d_n)
            tf.summary.scalar("generator_pos_loss", self.loss_g_p)
            tf.summary.scalar("generator_neg_loss", self.loss_g_n)

        var_ = tf.trainable_variables()
        self.var_d_p = [var for var in var_ if var.name == "dis_pos"]
        self.var_d_n = [var for var in var_ if var.name == "dis_neg"]
        self.var_g_p = [var for var in var_ if var.name == "converter_neg2pos"]
        self.var_g_n = [var for var in var_ if var.name == "converter_pos2neg"]

    def def_cell(self):
        if self.args.cell_model == 'rnn':
            cell_fn = tf.contrib.rnn.BasicRNNCell
        elif self.args.cell_model == 'gru':
            cell_fn = tf.contrib.rnn.GRUCell
        elif self.args.cell_model == 'lstm':
            cell_fn = tf.contrib.rnn.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(self.args.cell_model))

        
        cell_ = cell_fn(self.args.rnn_size, reuse=tf.get_variable_scope().reuse)
        if self.args.keep_prob < 1.:
            cell_ = tf.contrib.rnn.DropoutWrapper(cell_, output_keep_prob=self.args.keep_prob)
        return cell_

    def decoder(self, cell,  state):
        current_step= 0
        outputs = []
        embedding_weight = tf.Variable(tf.random_uniform([self.args.vocab_size, self.args.embedding_size],-1.,1.), name='embedding_weight')
        while current_step < self.args.max_time_step:
            if current_step != 0: 
                tf.get_variable_scope().reuse_variables()
                embedded = tf.reshape(tf.nn.embedding_lookup(embedding_weight, outputs[-1]), (-1, self.args.embedding_size))
            else:
                idx = tf.reshape(tf.convert_to_tensor([self.args.vocab_size-1]*self.args.batch_size), (-1, 1)) 
                embedded = tf.reshape(tf.nn.embedding_lookup(embedding_weight, idx), (-1, self.args.embedding_size))
        
            output, new_state = cell(embedded, state)
            outputs.append(tf.to_int32(tf.argmax(tf.layers.dense(output, self.args.vocab_size, tf.nn.softmax, name= "decoder_dense"), axis=-1)))
            current_step += 1
        return tf.transpose(tf.expand_dims(tf.convert_to_tensor(outputs), axis=-1),(1,0,2))
            
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
                
                for t in range(self.args.max_time_step):
                    embedded = tf.nn.embedding_lookup(embedding_weight, x[:,t,:])
                    rnn_inputs.append(embedded)
                rnn_inputs = tf.reshape(tf.transpose(tf.convert_to_tensor(rnn_inputs), (0,1,3,2)), (-1, self.args.max_time_step, self.args.embedding_size))
                print(rnn_inputs.get_shape().as_list())
            with tf.variable_scope('Encoder') as scope:
                encoder_cell = self.def_cell()     
                _, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, rnn_inputs, initial_state=encoder_cell.zero_state(batch_size=self.args.batch_size, dtype=tf.float32), dtype=tf.float32)

            with tf.variable_scope("Decoder") as scope:
                decoder_cell = self.def_cell()
                outputs = self.decoder(decoder_cell, encoder_final_state)

            return outputs

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

                    embedded = tf.nn.embedding_lookup(embedding_weight, x[:,t,:])
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
        opt_d_p = tf.train.AdamOptimizer(self.args.lr).minimize(self.loss_d_p, var_list=self.var_d_p)
        opt_d_n = tf.train.AdamOptimizer(self.args.lr).minimize(self.loss_d_n, var_list=self.var_d_n)
        opt_g_p = tf.train.AdamOptimizer(self.args.lr).minimize(self.loss_g_p, var_list=self.var_g_p)
        opt_g_n = tf.train.AdamOptimizer(self.args.lr).minimize(self.loss_g_n, var_list=self.var_g_n)

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
                    saver.save(sess, "save/model.ckpt")
                    print("----------------------saved model-------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--lr", dest="lr", type=float, default= 0.2)
    parser.add_argument("--data_dir", dest="data_dir", default="../data/")
    parser.add_argument("--cell_model", dest="cell_model", type=str, default="gru")
    parser.add_argument("--l1_lambda", dest="l1_lambda", type=float, default=50)
    parser.add_argument("--rnn_size", dest="rnn_size", type=int, default=1024)
    parser.add_argument("--index_dir", dest="index_dir", default="../data/index.txt")
    parser.add_argument("--itrs", dest="itrs", type=int, default=10001)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=10)
    parser.add_argument("--num_layers", dest="num_layers", type=int, default=1)
    parser.add_argument("--embedding_size", dest="embedding_size", default=64)
    parser.add_argument("--max_time_step", dest="max_time_step", type=int, default=20)
    parser.add_argument("--vocab_size", dest="vocab_size", type=int, default=2348)
    parser.add_argument("--train", dest="train", type=bool, default=True)
    parser.add_argument("--kernels", dest="kernels", type=list, default=[2,3,4,5,6])
    parser.add_argument("--keep_prob", dest="keep_prob", type=float, default=0.4)
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

