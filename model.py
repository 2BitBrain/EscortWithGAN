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

        self.A_inps = tf.placeholder(dtype=dtype, shape=shape)
        self.B_inps = tf.placeholder(dtype=dtype, shape=shape)
        self.go = tf.placeholder(dtype=dtype, shape=[None,  1] if args.embedding  else [None, args.vocab_size+2])

        self.A_pretrain_d_input = tf.placeholder(dtype=dtype, shape=shape)
        self.A_pretrain_label = tf.placeholder(dtype=tf.float32, shape=[None, args.max_time_step, args.vocab_size+2])
        
        self.B_pretrain_d_input = tf.placeholder(dtype=dtype, shape=shape)
        self.B_pretrain_label = tf.placeholder(dtype=tf.float32, shape=[None, args.max_time_step, args.vocab_size+2])
        
       #####start pre training#####
        A2A, regu_A_loss, A_e_cell, A_d_cell = generator(self.A_inps, self.A_pretrain_d_input, None, None, None, args, "g_A2B", False, False, True) 
        B2B, regu_B_loss, B_e_cell, B_d_cell = generator(self.B_inps, self.B_pretrain_d_input, None, None, None, args, "g_B2A", False, True, True)

        self.p_A_loss = tf.reduce_mean(tf.squared_difference(A2A, self.A_pretrain_label))# + regu_A_loss
        self.p_B_loss = tf.reduce_mean(tf.squared_difference(B2B, self.B_pretrain_label))# + regu_B_loss

       #####end pre training #### 

       #####start training #####
        self.A2B,_, A_e_cell, A_d_cell = generator(self.A_inps, None, self.go, A_e_cell, A_d_cell, args, "g_A2B", True, True, False)
        self.B2A,_, B_e_cell, B_d_cell = generator(self.B_inps, None, self.go, B_e_cell, B_d_cell, args, "g_B2A", True, True, False)

        cyc_inp_A2B = tf.expand_dims(tf.arg_max(self.A2B, 2), -1) if args.embedding else self.A2B
        B2A_,_,_,_ = generator(cyc_inp_A2B, None, self.go, B_e_cell, B_d_cell, args, "g_B2A", True, True, False)

        cyc_inp_B2A = tf.expand_dims(tf.arg_max(self.B2A, 2), -1) if args.embedding else self.B2A
        A2B_,_,_,_ = generator(cyc_inp_B2A, None, self.go, A_e_cell, A_d_cell, args, "g_A2B", True, True, False)

        dis_A_real, A_cell = discriminator(self.A_inps, None, args, "discriminator_A", False)
        dis_B_real, B_cell = discriminator(self.B_inps, None, args, "discriminator_B", False)
        
        dis_A_fake, _ = discriminator(cyc_inp_B2A, A_cell, args, "discriminator_A", True)
        dis_B_fake, _ = discriminator(cyc_inp_A2B, B_cell, args, "discriminator_B", True)

        loss_d_A = tf.reduce_mean(tf.square(dis_A_real-1)) + tf.reduce_mean(tf.square(dis_A_fake))
        loss_d_B = tf.reduce_mean(tf.square(dis_B_real-1)) + tf.reduce_mean(tf.square(dis_B_fake))
        self.d_loss = (loss_d_A + loss_d_B)/2
        
        A_inps = tf.one_hot(tf.squeeze(self.A_inps), args.vocab_size+2, 1., 0., -1) if args.embedding else self.A_inps
        B_inps = tf.one_hot(tf.squeeze(self.B_inps), args.vocab_size+2, 1., 0., -1) if args.embedding else self.B_inps
        cycle_loss = args.l_lambda * (tf.reduce_mean(tf.square(tf.abs(A_inps - B2A_))) + args.l_lambda * tf.reduce_mean(tf.square(tf.abs(B_inps - A2B_))))

        loss_g_A = tf.reduce_mean(tf.square(dis_A_fake-1))
        loss_g_B = tf.reduce_mean(tf.square(dis_B_fake-1))
        self.g_loss = (loss_g_B + loss_g_A) / 2 + cycle_loss

        #####end training #####
        
        with tf.variable_scope("summary") as scope:
            tf.summary.scalar("discriminator_A_loss", loss_d_A)
            tf.summary.scalar("discriminator_B_loss", loss_d_B)
            tf.summary.scalar("generator_A_loss", loss_g_A)
            tf.summary.scalar("generator_B_loss", loss_g_B)

        var_ = tf.trainable_variables()
        self.var_d_A = [var for var in var_ if  "discriminator_A" in var.name]
        self.var_d_B = [var for var in var_ if  "discriminator_B" in var.name]
        self.var_g_A = [var for var in var_ if  "g_B2A" in var.name]
        self.var_g_B = [var for var in var_ if  "g_A2B" in var.name]
        self.var_d = self.var_d_B + self.var_d_A
        self.var_g = self.var_g_B + self.var_g_A
        
    def train(self):
        opt_p_A = tf.train.GradientDescentOptimizer(self.args.lr).minimize(self.p_A_loss, var_list=self.var_g_B)
        opt_p_B = tf.train.GradientDescentOptimizer(self.args.lr).minimize(self.p_B_loss, var_list=self.var_g_A)
        opt_g = tf.train.GradientDescentOptimizer(self.args.g_lr).minimize(self.g_loss, var_list=self.var_g)
        opt_d = tf.train.GradientDescentOptimizer(self.args.d_lr).minimize(self.d_loss, var_list=self.var_d)

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
                    A_feed = {
                        self.A_inps: in_pos[choiced_pos_idx],
                        self.A_pretrain_d_input: d_in_pos[choiced_pos_idx],
                        self.A_pretrain_label: d_label_pos[choiced_pos_idx]
                    }                   

                    B_feed = {
                        self.B_inps: in_neg[choiced_neg_idx],
                        self.B_pretrain_d_input: d_in_neg[choiced_neg_idx],
                        self.B_pretrain_label: d_label_neg[choiced_neg_idx]
                    }

                    A_loss, _ = sess.run([self.p_A_loss, opt_p_A], A_feed)
                    B_loss, _ = sess.run([self.p_B_loss, opt_p_B], B_feed)

                    if i % 30 == 0:print("A_loss:", A_loss,"   B_loss:", B_loss)
                    if i % 60 == 0:p_saver.save(sess, self.args.pre_train_path+"model.ckpt")
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

                feed_dict = {self.A_inps:in_pos[pos_choiced_idx],
                             self.B_inps:in_neg[neg_choiced_idx],
                             self.go:go}

                _, loss_g = sess.run([opt_g, self.g_loss], feed_dict=feed_dict)
                _, loss_d = sess.run([opt_d, self.d_loss], feed_dict=feed_dict)

                if itr % 100 == 0: 
                    B_s, A_s = sess.run([self.A2B, self.B2A], feed_dict)
                    #visualizer(neg_s, pos_one_hot_sentences[pos_choiced_idx], "data/index.txt", "visualize_neg.txt")
                    #visualizer(pos_s, neg_one_hot_sentences[neg_choiced_idx],"data/index.txt", "visualize_pos.txt")
                    print("itr", itr, "loss_g", loss_g, "loss_d", loss_d)

                if itr % 10000 == 0:
                    saver_.save(sess, "saved/model.ckpt")
                    print("----------------------saved model-------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--lr", dest="lr", type=float, default= 0.1)
    parser.add_argument("--g_lr", dest="g_lr", type=float, default=0.08)
    parser.add_argument("--d_lr", dest="d_lr", type=float, default=0.01)
    parser.add_argument("--data_dir", dest="data_dir", default="../data/")
    parser.add_argument("--cell_model", dest="cell_model", type=str, default="lstm")
    parser.add_argument("--l1_lambda", dest="l1_lambda", type=float, default=50)
    parser.add_argument("--index_dir", dest="index_dir", default="../data/index.txt")
    parser.add_argument("--itrs", dest="itrs", type=int, default=1000001)
    parser.add_argument("--p_itrs", dest="p_itrs", type=int, default=10000)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=4)
    parser.add_argument("--embedding_size", dest="embedding_size", default=256)
    parser.add_argument("--rnn_embedding_size", dest="rnn_embedding_size", type=int, default=64)
    parser.add_argument("--max_time_step", dest="max_time_step", type=int, default=20)
    parser.add_argument("--vocab_size", dest="vocab_size", type=int, default=2346)
    parser.add_argument("--train", dest="train", type=bool, default=True)
    parser.add_argument("--keep_prob", dest="keep_prob", type=float, default=1)
    parser.add_argument("--gen_rnn_size", dest="gen_rnn_size", type=int, default=512)
    parser.add_argument("--dis_rnn_size", dest="dis_rnn_size", type=int, default=576)
    parser.add_argument("--merged_all", dest="merged_all", type=bool, default=False)
    parser.add_argument("--embedding", dest="embedding", type=bool, default=True)
    parser.add_argument("--scale", dest="scale", type=float, default=1.)  
    parser.add_argument("--use_extracted_feature", dest="use_extracted_feature", type=bool, default=False)
    parser.add_argument("--reg_constant", dest="reg_constant", type=float, default=1.)
    parser.add_argument("--l_labmda", dest="l_lambda", type=float, default=1.)
    parser.add_argument("--pre_train", dest="pre_train", type=bool, default=True)
    parser.add_argument("--pre_train_done", dest="pre_train_done", type=bool, default=False)
    parser.add_argument("--num_g_layers", dest="num_g_layers", type=int, default=2)
    parser.add_argument("--pre_train_path", dest="pre_train_path", type=str, default="pre_train_saved/")
    args= parser.parse_args()
    
    if not os.path.exists("saved"):
        os.mkdir("saved")

    if not os.path.exists("logs"):
        os.mkdir("logs")

    if not os.path.exists(args.pre_train_path):
        os.mkdir(args.pre_train_path)

    model_ = model(args)
    if args.train:
        model_.train()

