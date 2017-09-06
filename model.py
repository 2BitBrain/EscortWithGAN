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
        self.go = tf.placeholder(dtype=dtype, shape=[None,  1] if args.embedding  else [None, args.vocab_size])

        self.pos_pretrain_e_input = tf.placeholder(dtype=dtype, shape=shape)
        self.pos_pretrain_d_input = tf.placeholder(dtype=dtype, shape=shape)
        self.pos_pretrain_label = tf.placeholder(dtype=dtype, shape=shape)
        
        self.neg_pretrain_e_input = tf.placeholder(dtype=dtype, shape=shape)
        self.neg_pretrain_d_input = tf.placeholder(dtype=dtype, shape=shape)
        self.neg_pretrain_label = tf.placeholder(dtype=dtype, shape=shape)
        
       #####start pre training#####
        pos2pos, regu_p_loss, p_e_cell, p_d_cell = generator(self.pos_pretrain_e_input, self.pos_pretrain_d_input, None, None, None, args, "g_pos2neg", False, False, True) 
        neg2neg, regu_n_loss, n_e_cell, n_d_cell = generator(self.neg_pretrain_e_input, self.neg_pretrain_d_input, None, None, None, args, "g_neg2pos", False, True, True)

        self.p_p_loss = tf.squared_difference(pos2pos, self.pos_pretrain_label) + regu_p_loss
        self.p_n_loss = tf.squared_difference(neg2neg, self.neg_pretrain_label) + regu_n_loss

       #####end pre training #### 

       #####start training #####
        self.pos2neg,_, p_e_cell, p_d_cell = generator(self.pos_inps, None, self.go, p_e_cell, p_d_cell,args, "g_pos2neg", True, True, False)
        self.neg2pos,_, n_e_cell, n_d_cell = generator(self.neg_inps, None, self.go, n_e_cell, n_d_cell, args, "g_neg2pos", True, True, False)
        neg2pos_,_,_,_ = generator(self.pos2neg, None, self.go, n_e_cell, n_d_cell, args, "g_neg2pos", True, True, False)
        pos2neg_,_,_,_ = generator(self.neg2pos, None, self.go, p_e_cell, p_d_cell, args, "g_pos2neg", True, True, False)

        dis_p_real, p_cell = discriminator(self.pos_inps, None, args, "discriminator_pos", False)
        dis_n_real, n_cell = discriminator(self.neg_inps, None, args, "discriminator_neg", False)
        dis_p_fake, _ = discriminator(self.neg2pos, p_cell, args, "discriminator_pos", True)
        dis_n_fake, _ = discriminator(self.pos2neg, n_cell, args, "discriminator_neg", True)

        loss_d_p = tf.reduce_mean(tf.square(1-dis_p_real)) + tf.reduce_mean(tf.square(dis_p_fake))
        loss_d_n = tf.reduce_mean(tf.square(1-dis_n_real)) + tf.reduce_mean(tf.square(dis_n_fake))
        self.d_loss = (loss_d_n + loss_d_p)/2

        if not args.embedding:
            cycle_loss = args.l_lambda * (tf.reduce_mean(tf.square(tf.abs(self.pos_inps - neg2pos_))) + tf.reduce_mean(tf.square(tf.abs(self.neg_inps - pos2neg_))))
        else:
            cycle_loss = 0.

        loss_g_p = tf.reduce_mean(tf.square(1 - dis_p_fake))
        loss_g_n = tf.reduce_mean(tf.square(1 - dis_n_fake))
        self.g_loss = (loss_g_n + loss_g_p) / 2 + cycle_loss

        #####end training #####
        
        with tf.variable_scope("summary") as scope:
            tf.summary.scalar("discriminator_pos_loss", loss_d_p)
            tf.summary.scalar("discriminator_neg_loss", loss_d_n)
            tf.summary.scalar("generator_pos_loss", loss_g_p)
            tf.summary.scalar("generator_neg_loss", loss_g_n)

        var_ = tf.trainable_variables()
        self.var_d_p = [var for var in var_ if  "discriminator_pos" in var.name]
        self.var_d_n = [var for var in var_ if  "discriminator_neg" in var.name]
        self.var_g_p = [var for var in var_ if  "g_neg2pos" in var.name]
        self.var_g_n = [var for var in var_ if  "g_pos2neg" in var.name]
        self.var_d = self.var_d_n + self.var_d_p
        self.var_g = self.var_g_n + self.var_g_n
        
    def train(self):
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
            
            

            for itr in range(self.args.itrs):
                pos_choiced_idx = random.sample(range(pos_data_size), self.args.batch_size)
                neg_choiced_idx = random.sample(range(neg_data_size), self.args.batch_size)

                feed_dict = {self.pos_inps:pos_one_hot_sentences[pos_choiced_idx],self.pos_inps_indexs:pos_converted_sentences[pos_choiced_idx], self.neg_inps:neg_one_hot_sentences[neg_choiced_idx], self.neg_inps_indexs:neg_converted_sentences[neg_choiced_idx], self.go:go}
                

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
    parser.add_argument("--lr", dest="lr", type=float, default= 0.1)
    parser.add_argument("--data_dir", dest="data_dir", default="../data/")
    parser.add_argument("--cell_model", dest="cell_model", type=str, default="gru")
    parser.add_argument("--l1_lambda", dest="l1_lambda", type=float, default=50)
    parser.add_argument("--index_dir", dest="index_dir", default="../data/index.txt")
    parser.add_argument("--itrs", dest="itrs", type=int, default=1000001)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=4)
    parser.add_argument("--embedding_size", dest="embedding_size", default=64)
    parser.add_argument("--rnn_embedding_size", dest="rnn_embedding_size", type=int, default=64)
    parser.add_argument("--max_time_step", dest="max_time_step", type=int, default=20)
    parser.add_argument("--vocab_size", dest="vocab_size", type=int, default=2348)
    parser.add_argument("--train", dest="train", type=bool, default=True)
    parser.add_argument("--keep_prob", dest="keep_prob", type=float, default=1)
    parser.add_argument("--gen_rnn_size", dest="gen_rnn_size", type=int, default=1024)
    parser.add_argument("--dis_rnn_size", dest="dis_rnn_size", type=int, default=576)
    parser.add_argument("--merged_all", dest="merged_all", type=bool, default=False)
    parser.add_argument("--embedding", dest="embedding", type=bool, default=False)
    parser.add_argument("--scale", dest="scale", type=float, default=1.)  
    parser.add_argument("--use_extracted_feature", dest="use_extracted_feature", type=bool, default=False)
    parser.add_argument("--decoder_embedding", dest="decoder_embedding", type=bool, default=False)
    parser.add_argument("--reg_constant", dest="reg_constant", type=float, default=1.)
    parser.add_argument("--l_labmda", dest="l_lambda", type=float, default=1.)
    parser.add_argument("--pre_train", dest="pre_train", type=bool, default=True)
    parser.add_argument("--pre_train_done", dest="pre_train_done", type=bool, default=False)
    args= parser.parse_args()
    
    if not os.path.exists("save"):
        os.mkdir("save")

    if not os.path.exists("logs"):
        os.mkdir("logs")

    model_ = model(args)
    if args.train:
        model_.train()

