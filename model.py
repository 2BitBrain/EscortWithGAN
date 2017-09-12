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
            shape = [None, args.max_time_step, args.vocab_size+2]
            dtype=tf.float32

        self.pos_inps = tf.placeholder(dtype=dtype, shape=shape)
        self.neg_inps = tf.placeholder(dtype=dtype, shape=shape)
        self.go = tf.placeholder(dtype=dtype, shape=[None,  1] if args.embedding  else [None, args.vocab_size+2])

        self.pos_pretrain_d_input = tf.placeholder(dtype=dtype, shape=shape)
        self.pos_pretrain_label = tf.placeholder(dtype=tf.float32, shape=[None, args.max_time_step, args.vocab_size+2])
        
        self.neg_pretrain_d_input = tf.placeholder(dtype=dtype, shape=shape)
        self.neg_pretrain_label = tf.placeholder(dtype=tf.float32, shape=[None, args.max_time_step, args.vocab_size+2])
        
       #####start pre training#####
        pos2pos, regu_p_loss, p_e_cell, p_d_cell = generator(self.pos_inps, self.pos_pretrain_d_input, None, None, None, args, "g_pos2neg", False, False, True) 
        neg2neg, regu_n_loss, n_e_cell, n_d_cell = generator(self.neg_inps, self.neg_pretrain_d_input, None, None, None, args, "g_neg2pos", False, True, True)

        self.p_p_loss = tf.squared_difference(pos2pos, self.pos_pretrain_label) + regu_p_loss
        self.p_n_loss = tf.squared_difference(neg2neg, self.neg_pretrain_label) + regu_n_loss

       #####end pre training #### 

       #####start training #####
        self.pos2neg,_, p_e_cell, p_d_cell = generator(self.pos_inps, None, self.go, p_e_cell, p_d_cell,args, "g_pos2neg", True, True, False)
        self.neg2pos,_, n_e_cell, n_d_cell = generator(self.neg_inps, None, self.go, n_e_cell, n_d_cell, args, "g_neg2pos", True, True, False)
        
        print(self.pos2neg.dtype)

        cyc_inp_p2n = tf.expand_dims(tf.arg_max(self.pos2neg, 2), -1) if args.embedding else self.pos2neg
        neg2pos_,_,_,_ = generator(cyc_inp_p2n, None, self.go, n_e_cell, n_d_cell, args, "g_neg2pos", True, True, False)

        cyc_inp_n2p = tf.expand_dims(tf.arg_max(self.neg2pos, 2), -1) if args.embedding else self.neg2pos
        pos2neg_,_,_,_ = generator(cyc_inp_n2p, None, self.go, p_e_cell, p_d_cell, args, "g_pos2neg", True, True, False)

        dis_p_real, p_cell = discriminator(self.pos_inps, None, args, "discriminator_pos", False)
        dis_n_real, n_cell = discriminator(self.neg_inps, None, args, "discriminator_neg", False)
        
        dis_p_fake, _ = discriminator(cyc_inp_n2p, p_cell, args, "discriminator_pos", True)
        dis_n_fake, _ = discriminator(cyc_inp_p2n, n_cell, args, "discriminator_neg", True)

        loss_d_p = tf.reduce_mean(tf.square(1-dis_p_real)) + tf.reduce_mean(tf.square(dis_p_fake))
        loss_d_n = tf.reduce_mean(tf.square(1-dis_n_real)) + tf.reduce_mean(tf.square(dis_n_fake))
        self.d_loss = (loss_d_n + loss_d_p)/2
        
        pos_inps = tf.one_hot(tf.squeeze(self.pos_inps), args.vocab_size+2, 1., 0., -1) if args.embedding else self.pos_inps
        neg_inps = tf.one_hot(tf.squeeze(self.neg_inps), args.vocab_size+2, 1., 0., -1) if args.embedding else self.neg_inps
        cycle_loss = args.l_lambda * (tf.reduce_mean(tf.square(tf.abs(pos_inps - neg2pos_))) + tf.reduce_mean(tf.square(tf.abs(neg_inps - pos2neg_))))

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
        self.var_g = self.var_g_n + self.var_g_p
        
    def train(self):
        opt_p_p = tf.train.GradientDescentOptimizer(self.args.lr).minimize(self.p_p_loss, var_list=self.var_g_n)
        opt_p_n = tf.train.GradientDescentOptimizer(self.args.lr).minimize(self.p_n_loss, var_list=self.var_g_p)
        opt_g = tf.train.GradientDescentOptimizer(self.args.lr).minimize(self.g_loss, var_list=self.var_g)
        opt_d = tf.train.GradientDescentOptimizer(self.args.lr).minimize(self.d_loss, var_list=self.var_d)

        mk_pre_train_func, mk_train_func = mk_train_data("./data/train.txt", "./data/index.txt", self.args.max_time_step, self.args.embedding)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES, scope='Word_Level_CNN'))
            saver.restore(sess, "./word_level_cnn_save/word_level_cnn_model.ckpt")
            saver_ = tf.train.Saver(tf.global_variables())
            p_saver = tf.train.Saver(tf.global_variables())
            graph = tf.summary.FileWriter('./logs', sess.graph)
            merged_summary = tf.summary.merge_all()
            
            ## Part of Pre-Training ##
            if self.args.pre_train and not self.args.pre_train_done:
                in_neg, d_in_neg, d_label_neg, in_pos, d_in_pos, d_label_pos = mk_pre_train_func()
                neg_ = range(in_neg.shape[0])
                pos_ = range(in_pos.shape[0])
                print("## start to pre train ##")
                for i in range(self.args.p_itrs):
                    choiced_pos_idx = [random.choice(pos_) for _ in range(self.args.batch_size)] 
                    choiced_neg_idx = [random.choice(neg_) for _ in range(self.args.batch_size)]
                    pos_feed = {
                        self.pos_inps: in_pos[choiced_pos_idx],
                        self.pos_pretrain_d_input: d_in_pos[choiced_pos_idx],
                        self.pos_pretrain_label: d_label_pos[choiced_pos_idx]
                    }                   

                    neg_feed = {
                        self.neg_inps: in_neg[choiced_neg_idx],
                        self.neg_pretrain_d_input: d_in_neg[choiced_neg_idx],
                        self.neg_pretrain_label: d_label_neg[choiced_neg_idx]
                    }

                    p_loss, _ = sess.run([self.p_p_loss, opt_p_p], pos_feed)
                    n_loss, _ = sess.run([self.p_n_loss, opt_p_n], neg_feed)

                    if i % 30 == 0:print("p_loss:", p_loss,"   n_loss:", n_loss)
                    if i % 60 == 0:p_saver.save(sess, self.args.pre_train_path)
                print("## pre training done ! ##")

            elif self.args.pre_train:
                if not os.path.exits(self.args.pre_train_path):
                    print("trained model file does not exits")
                    return 
                
                p_saver.restore(sess, self.args.pre_train_path)
                print("## restore done ! ##")
            
            ## Training Network part of all ##
            in_neg, in_pos = mk_train_func()
            go = mk_go(self.args.batch_size, self.args.vocab_size, self.args.embedding)
            pos_range = range(in_pos.shape[0])
            neg_range = range(in_neg.shape[0])
            for itr in range(self.args.itrs):
                pos_choiced_idx = random.sample(pos_range, self.args.batch_size)
                neg_choiced_idx = random.sample(neg_range, self.args.batch_size)

                feed_dict = {self.pos_inps:in_pos[pos_choiced_idx],
                             self.neg_inps:in_neg[neg_choiced_idx],
                             self.go:go}

                _, loss_g = sess.run([opt_g, self.g_loss], feed_dict=feed_dict)
                _, loss_d = sess.run([opt_d, self.d_loss], feed_dict=feed_dict)

                if itr % 100 == 0: 
                    neg_s, pos_s = sess.run([self.pos2neg, self.neg2pos], feed_dict)
                    #visualizer(neg_s, pos_one_hot_sentences[pos_choiced_idx], "data/index.txt", "visualize_neg.txt")
                    #visualizer(pos_s, neg_one_hot_sentences[neg_choiced_idx],"data/index.txt", "visualize_pos.txt")
                    print("itr", itr, "loss_g", loss_g, "loss_d", loss_d)

                if itr % 10000 == 0:
                    saver_.save(sess, "saved/model.ckpt")
                    print("----------------------saved model-------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--lr", dest="lr", type=float, default= 0.1)
    parser.add_argument("--data_dir", dest="data_dir", default="../data/")
    parser.add_argument("--cell_model", dest="cell_model", type=str, default="gru")
    parser.add_argument("--l1_lambda", dest="l1_lambda", type=float, default=50)
    parser.add_argument("--index_dir", dest="index_dir", default="../data/index.txt")
    parser.add_argument("--itrs", dest="itrs", type=int, default=1000001)
    parser.add_argument("--p_itrs", dest="p_itrs", type=int, default=10000)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=4)
    parser.add_argument("--embedding_size", dest="embedding_size", default=64)
    parser.add_argument("--rnn_embedding_size", dest="rnn_embedding_size", type=int, default=64)
    parser.add_argument("--max_time_step", dest="max_time_step", type=int, default=20)
    parser.add_argument("--vocab_size", dest="vocab_size", type=int, default=2346)
    parser.add_argument("--train", dest="train", type=bool, default=True)
    parser.add_argument("--keep_prob", dest="keep_prob", type=float, default=1)
    parser.add_argument("--gen_rnn_size", dest="gen_rnn_size", type=int, default=1024)
    parser.add_argument("--dis_rnn_size", dest="dis_rnn_size", type=int, default=576)
    parser.add_argument("--merged_all", dest="merged_all", type=bool, default=False)
    parser.add_argument("--embedding", dest="embedding", type=bool, default=True)
    parser.add_argument("--scale", dest="scale", type=float, default=1.)  
    parser.add_argument("--use_extracted_feature", dest="use_extracted_feature", type=bool, default=False)
    parser.add_argument("--reg_constant", dest="reg_constant", type=float, default=1.)
    parser.add_argument("--l_labmda", dest="l_lambda", type=float, default=1.)
    parser.add_argument("--pre_train", dest="pre_train", type=bool, default=False)
    parser.add_argument("--pre_train_done", dest="pre_train_done", type=bool, default=False)
    args= parser.parse_args()
    
    if not os.path.exists("saved"):
        os.mkdir("saved")

    if not os.path.exists("logs"):
        os.mkdir("logs")

    model_ = model(args)
    if args.train:
        model_.train()

