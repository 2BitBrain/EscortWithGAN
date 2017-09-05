import sys
sys.path.append('../sentiment_analysis/')

import argparse
import tensorflow as tf
import numpy as np
import tensorflow.contrib.seq2seq as seq2seq
from util import *
from module import *
import os
import random

class model():
    def __init__(self, args):
        self.args = args 
        
        ## Define feed items ##
        # if args.embedding    #
        #this model use tensor-#
        #flow embeddign_look_-#
        #up. and dtype is int32#
        #                       #
        ###################

        """
        @param  pos_inps Feeded item for generator which is converting pos2neg. 
        @param  neg_inps Feeded item for generator which is converting neg2pos.
        @param  go feeded item for generator which is initial decoder part.
        @param (pos_pretrain_input&neg_pretrain_input) Feeded itemf for pre training generator
        """

        if args.embedding:
            shape = [None, args.max_time_step, 1]
            dtype = tf.int32
        else:
            shape = [None, args.max_time_step, args.vocab_size]
            dtype=tf.float32

        self.pos_inps = tf.placeholder(dtype=dtype, shape=shape)
        self.neg_inps = tf.placeholder(dtype=dtype, shape=shape)
        self.go = tf.placeholder(dtype=dtype, shape=[None, args.max_time_step, 1] if args.embedding [None, args.max_time_step, args.vocab_size])

        self.pos_pretrain_e_input = tf.placeholder(dtype=dtype, shape=shape)
        self.pos_pretrain_d_input = tf.placeholder(dtype=dtype, shape=shape)
        self.pos_pretrain_label = tf.placeholder(dtype=dtype, shape=shape)
        
        self.neg_pretrain_e_input = tf.placeholder(dtype=dtype, shape=shape)
        self.neg_pretrain_d_input = tf.placeholder(dtype=dtype, shape=shape)
        self.neg_pretrain_label = tf.placeholder(dtype=dtype, shape=shape)
        
       #####start pre training#####
        pos2pos, regu_p_loss = generator(self.pos_pretrain_e_input, self.pos_pretrain_d_input, None, args, "g_pos2neg", False, False, True) 
        neg2neg, regu_n_loss = generator(self.neg_pretrain_e_input, self.neg_pretrain_d_input, None, args, "g_neg2pos", False, True, True)

        self.p_p_loss = tf.squared_difference(pos2pos, self.pos_pretrain_label) + regu_p_loss
        self.p_n_loss = tf.squared_difference(neg2neg, self.neg_pretrain_label) + regu_n_loss

       #####end pre training #### 

       #####start training #####
        self.pos2neg = generator(self.pos_inps, None, self.go, args, "g_pos2neg", True, True, False)
        self.neg2pos = generator(self.neg_inps, None, self.go, args, "g_neg2pos", True, True, True)
        neg2pos_ = generator(self.pos2neg, None, self.go, args, "g_neg2pos", True, True, False)
        pos2neg_ = generator(self.neg2pos, None, self.go, args, "g_pos2neg", True, True, False)

        

       #####end training #####

        self.d_loss = self.loss_d_n + self.loss_d_p

        self.loss_g_p = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_pos, labels=tf.ones_like(dis_fake_pos))) #+ args.l1_lambda*tf.reduce_mean(tf.abs(self.pos_inps - neg_pos_outs))
        self.loss_g_n = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_neg, labels=tf.ones_like(dis_fake_neg))) #+ args.l1_lambda*tf.reduce_mean(tf.abs(self.neg_inps - pos_neg_outs))

        self.g_loss = self.loss_g_n + self.loss_g_p# + l_n + l_p

        with tf.variable_scope("summary") as scope:
            tf.summary.scalar("discriminator_pos_loss", self.loss_d_p)
            tf.summary.scalar("discriminator_neg_loss", self.loss_d_n)
            tf.summary.scalar("generator_pos_loss", self.loss_g_p)
            tf.summary.scalar("generator_neg_loss", self.loss_g_n)

        var_ = tf.trainable_variables()
        self.var_d_p = [var for var in var_ if  "dis_pos" in var.name]
        self.var_d_n = [var for var in var_ if  "dis_neg" in var.name]
        self.var_g_p = [var for var in var_ if  "converter_neg2pos" in var.name]
        self.var_g_n = [var for var in var_ if  "converter_pos2neg" in var.name]
        self.var_d = [var for var in var_ if "dis" in var.name]
        self.var_g = [var for var in var_ if "converter" in var.name]
        
    def train(self):
       # opt_d_p = tf.train.AdamOptimizer(self.args.lr).minimize(self.loss_d_p, var_list=self.var_d_p)
       # opt_d_n = tf.train.AdamOptimizer(self.args.lr).minimize(self.loss_d_n, var_list=self.var_d_n)
       # opt_g_p = tf.train.AdamOptimizer(self.args.lr).minimize(self.loss_g_p, var_list=self.var_g_p)
       # opt_g_n = tf.train.AdamOptimizer(self.args.lr).minimize(self.loss_g_n, var_list=self.var_g_n)
        opt_g = tf.train.GradientDescentOptimizer(self.args.lr).minimize(self.g_loss, var_list=self.var_g)
        opt_d = tf.train.GradientDescentOptimizer(self.args.lr).minimize(self.d_loss, var_list=self.var_d)

        neg_converted_sentences, neg_one_hot_sentences, pos_converted_sentences, pos_one_hot_sentences = mk_train_data("./data/train.txt", "./data/index.txt", self.args.max_time_step)
        neg_data_size = neg_converted_sentences.shape[0]
        pos_data_size = pos_converted_sentences.shape[0]
        
        go = mk_go(self.args.batch_size, self.args.vocab_size)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES, scope='Word_Level_CNN'))
            saver.restore(sess, "./word_level_cnn_save/word_level_cnn_model.ckpt")
            saver_ = tf.train.Saver(tf.global_variables())

            graph = tf.summary.FileWriter('./logs', sess.graph)
            merged_summary = tf.summary.merge_all()
            
            print([var.name for var in tf.trainable_variables()])

            for itr in range(self.args.itrs):
                pos_choiced_idx = random.sample(range(pos_data_size), self.args.batch_size)
                neg_choiced_idx = random.sample(range(neg_data_size), self.args.batch_size)

                feed_dict = {self.pos_inps:pos_one_hot_sentences[pos_choiced_idx],self.pos_inps_indexs:pos_converted_sentences[pos_choiced_idx], self.neg_inps:neg_one_hot_sentences[neg_choiced_idx], self.neg_inps_indexs:neg_converted_sentences[neg_choiced_idx], self.go:go}
                #_, loss_g_n = sess.run([opt_g_n, self.loss_g_n], feed_dict=feed_dict)
                #_, loss_d_n = sess.run([opt_d_n, self.loss_d_n], feed_dict=feed_dict)
                #_, loss_g_p = sess.run([opt_g_p, self.loss_g_p], feed_dict=feed_dict)
                #_, loss_d_p = sess.run([opt_d_p, self.loss_d_p], feed_dict=feed_dict)
                _, loss_g = sess.run([opt_g, self.g_loss], feed_dict=feed_dict)
                _, loss_d = sess.run([opt_d, self.d_loss], feed_dict=feed_dict)

                if itr % 100 == 0:
                    feed_dict = {self.pos_inps_indexs:pos_converted_sentences[pos_choiced_idx], self.neg_inps_indexs:neg_converted_sentences[neg_choiced_idx], self.go:go} 
                    neg_s, pos_s = sess.run([self.neg_outs, self.pos_outs], feed_dict)
                    visualizer(neg_s, pos_one_hot_sentences[pos_choiced_idx], "data/index.txt", "visualize_neg.txt")
                    visualizer(pos_s, neg_one_hot_sentences[neg_choiced_idx],"data/index.txt", "visualize_pos.txt")
                    print("itr", itr, "loss_g", loss_g, "loss_d", loss_d)

                if itr % 10000 == 0:
                    saver_.save(sess, "save/model.ckpt")
                    print("----------------------saved model-------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--lr", dest="lr", type=float, default= 0.002)
    parser.add_argument("--data_dir", dest="data_dir", default="../data/")
    parser.add_argument("--cell_model", dest="cell_model", type=str, default="gru")
    parser.add_argument("--l1_lambda", dest="l1_lambda", type=float, default=50)
    parser.add_argument("--index_dir", dest="index_dir", default="../data/index.txt")
    parser.add_argument("--itrs", dest="itrs", type=int, default=1000001)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=50)
    parser.add_argument("--embedding_size", dest="embedding_size", default=64)
    parser.add_argument("--rnn_embedding_size", dest="rnn_embedding_size", type=int, default=64)
    parser.add_argument("--max_time_step", dest="max_time_step", type=int, default=20)
    parser.add_argument("--vocab_size", dest="vocab_size", type=int, default=2348)
    parser.add_argument("--train", dest="train", type=bool, default=True)
    parser.add_argument("--keep_prob", dest="keep_prob", type=float, default=1)
    parser.add_argument("--gen_rnn_size", dest="gen_rnn_size", type=int, default=1024)
    parser.add_argument("--dis_rnn_size", dest="dis_rnn_size", type=int, default=576)
    parser.add_argument("--merged_all", dest="merged_all", type=bool, default=False)
    args= parser.parse_args()
    
    if not os.path.exists("save"):
        os.mkdir("save")

    if not os.path.exists("logs"):
        os.mkdir("logs")

    model_ = model(args)
    if args.train:
        model_.train()

