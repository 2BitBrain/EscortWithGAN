import sys
sys.path.append('../sentiment_analysis/')

import argparse
import tensorflow as tf
import numpy as np
import tensorflow.contrib.seq2seq as seq2seq
from util import *
from module import *
import os

class model():
    def __init__(self, args):
        self.args = args 
         
        self.pos_inps = tf.placeholder(dtype=tf.float32, shape=[None, args.max_time_step, args.vocab_size])
        self.pos_inps_indexs = tf.placeholder(dtype=tf.int32, shape=[None, args.max_time_step, 1])
        self.neg_inps = tf.placeholder(dtype=tf.float32, shape=[None, args.max_time_step, args.vocab_size])
        self.neg_inps_indexs = tf.placeholder(dtype=tf.int32, shape=[None, args.max_time_step, 1])

        converted_neg, self.neg_outs, neg_indexs = converter(self.pos_inps, self.pos_inps_indexs, args, "converter_pos2neg")
        converted_pos, self.pos_outs, pos_indexs = converter(self.neg_inps, self.neg_inps_indexs, args, "converter_neg2pos", extract_reuse=True)
       
        converted_neg_pos, neg_pos_outs, neg_pos_indexs = converter(converted_neg, neg_indexs, args, "converter_neg2pos", reuse=True, extract_reuse=True)
        converted_pos_neg, pos_neg_outs, pos_neg_indexs = converter(converted_pos, pos_indexs, args, "converter_pos2neg", reuse=True, extract_reuse=True)

        dis_pos = discriminator(self.pos_inps, args, "dis_pos")
        dis_fake_pos = discriminator(self.pos_outs, args, "dis_pos", reuse=True)

        dis_neg = discriminator(self.neg_inps, args, "dis_neg")
        dis_fake_neg = discriminator(self.neg_outs, args, "dis_neg", reuse=True)

        self.loss_d_p = -1*(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_pos, labels=tf.ones_like(dis_pos))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_pos, labels=tf.zeros_like(dis_fake_pos))))
        self.loss_d_n = -1*(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_neg, labels=tf.ones_like(dis_neg))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_neg, labels=tf.zeros_like(dis_fake_neg))))
        
        cycle_loss = tf.to_float(tf.reduce_mean(tf.abs(tf.subtract(converted_neg_pos, self.pos_inps)))) + tf.to_float(tf.reduce_mean(tf.abs(tf.subtract(converted_pos_neg, self.neg_inps)))) 

        self.loss_g_p = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_pos, labels=tf.ones_like(dis_fake_pos))) + args.l1_lambda*cycle_loss
        self.loss_g_n = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_neg, labels=tf.ones_like(dis_fake_neg))) + args.l1_lambda*cycle_loss

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
            saver.restore(sess, "./save/word_level_cnn_model.ckpt")
            
            graph = tf.summary.FileWriter('./logs', sess.graph)
            merged_summary = tf.summary.merge_all()

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
    parser.add_argument("--gen_rnn_size", dest="gen_rnn_size", type=int, default=576)
    parser.add_argument("--test", dest="test", type=bool, default=True)
    parser.add_argument("--dis_rnn_size", dest="dis_rnn_size", type=int, default=512)
    parser.add_argument("--merged_all", dest="merged_all", type=bool, default=True)
    args= parser.parse_args()
    
    if not os.path.exists("save"):
        os.mkdir("save")

    if not os.path.exists("logs"):
        os.mkdir("logs")

    model_ = model(args)
    if args.train:
        model_.train()

