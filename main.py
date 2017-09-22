import os
import argparse
from model import *
import sys 
from util import *

def check_args(args):
    ErrorMess = ""
    if not args.num_d_layers > 0:
        ErrorMess+= "Please chose number larger than 0\n"
    if args.dis_rnn_size%2 != 0:
        ErrorMess+= "args.dis_rnn_size Please chose even number\n"
    if args.gen_rnn_size%2 != 0:
        ErrorMess+="args.gen_rnn_size Please chose even number\n"
    if not os.path.exists(args.A_index_path):
        vocabs = mk_dict_from_wakatied(args.A_corpus_path)
        save_index(args.A_index_path, vocabs)
    if not os.path.exists(args.B_index_path):
        vocabs = mk_dict_from_wakatied(args.B_corpus_path)
        save_index(args.B_index_path, vocabs)
    if not os.path.exists(args.Marged_index_path):
        marged =marge_vocab(read_index(args.A_index_path), read_index(args.B_index_path))
        save_index(args.Marged_index_path, marged)
    if ErrorMess != "":
        print(ErrorMess)
        sys.exit()        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--lr", dest="lr", type=float, default= 0.0001)
    parser.add_argument("--g_lr", dest="g_lr", type=float, default=0.0001)
    parser.add_argument("--d_lr", dest="d_lr", type=float, default=0.00001)
    parser.add_argument("--data_dir", dest="data_dir", default="../data/")
    parser.add_argument("--cell_model", dest="cell_model", type=str, default="lstm")
    parser.add_argument("--l1_lambda", dest="l1_lambda", type=float, default=30)
    parser.add_argument("--index_dir", dest="index_dir", default="../data/index.txt")
    parser.add_argument("--itrs", dest="itrs", type=int, default=1000001)
    parser.add_argument("--p_itrs", dest="p_itrs", type=int, default=4001)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=1)
    parser.add_argument("--embedding_size", dest="embedding_size", default=312)
    parser.add_argument("--rnn_embedding_size", dest="rnn_embedding_size", type=int, default=64)
    parser.add_argument("--max_time_step", dest="max_time_step", type=int, default=20)
    parser.add_argument("--vocab_size", dest="vocab_size", type=int, default=68579)
    parser.add_argument("--train", dest="train", type=bool, default=True)
    parser.add_argument("--keep_prob", dest="keep_prob", type=float, default=0.3)
    parser.add_argument("--gen_rnn_size", dest="gen_rnn_size", type=int, default=512)
    parser.add_argument("--dis_rnn_size", dest="dis_rnn_size", type=int, default=576)
    parser.add_argument("--merged_all", dest="merged_all", type=bool, default=False)
    parser.add_argument("--embedding", dest="embedding", type=bool, default=True)
    parser.add_argument("--scale", dest="scale", type=float, default=1.)  
    parser.add_argument("--reg_constant", dest="reg_constant", type=float, default=1.)
    parser.add_argument("--l_labmda", dest="l_lambda", type=float, default=40.)
    parser.add_argument("--pre_train", dest="pre_train", type=bool, default=True)
    parser.add_argument("--pre_train_done", dest="pre_train_done", type=bool, default=True)
    parser.add_argument("--num_g_layers", dest="num_g_layers", type=int, default=2)
    parser.add_argument("--pre_train_path", dest="pre_train_path", type=str, default="pre_train_saved/")
    parser.add_argument("--num_d_layers", dest="num_d_layers", type=int, default=2)
    parser.add_argument("--attention_units", dest="attention_units", type=int, default=512)
    parser.add_argument("--A_corpus_path", dest="A_corpus_path", type=str, default="data/wakatied_twitter.txt")
    parser.add_argument("--A_index_path", dest="A_index_path", type=str, default="data/twitter_index.txt")
    parser.add_argument("--B_corpus_path", dest="B_corpus_path", type=str, default="data/wakatied_nov18.txt")
    parser.add_argument("--B_index_path", dest="B_index_path", type=str, default="data/nov18_index.txt")
    parser.add_argument("--Marged_index_path", dest="Marged_index_path", type=str, default="data/marged_index.txt") 
    args= parser.parse_args()
   
    check_args(args)

    if not os.path.exists("summary"):
        os.mkdir("summary")

    if not os.path.exists("saved"):
        os.mkdir("saved")

    if not os.path.exists("logs"):
        os.mkdir("logs")

    if not os.path.exists(args.pre_train_path):
        os.mkdir(args.pre_train_path)

    model_ = model(args)
    if args.train:
        add_summary("itr", "gen_loss", "dis_loss", "summary/glr_{}_dlr_{}".format(args.g_lr, args.d_lr))
        model_.train()
