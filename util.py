import os
import numpy as np
import random
import MeCab

def visualizer(x, y, index_path, save_path):
    indexs = read_index(index_path)
    indexs.append("")#<GO>
    indexs.append("")#<END>
    sentences_x = []
    sentences_y = []
    #print(x.shape, y.shape)
    y = np.reshape(y, y.shape[:-1]) 
    for s_x, s_y in zip(x, y):
        idxs_x = np.argmax(s_x, axis=-1)
        sentences_x.append(" ".join([indexs[idx] for idx in idxs_x]))
        sentences_y.append(" ".join([indexs[idx] for idx in s_y]))

    with open(save_path, "a") as fs:
        fs.write("".join(["{} | {}\n".format(x,y) for x,y in zip(sentences_x, sentences_y)]))

def add_summary(itr, gen_loss, dis_loss, filename):
    with open(filename, 'a') as fs:
        fs.write("{},{},{}\n".format(itr, gen_loss, dis_loss))

##this function reads corpus from read_path.
def read_corpus(read_path):
    with open(read_path) as fs:
        lines = fs.readlines()
    return [line.split("\n")[0] for line in lines]
    
# !!!wakati is used for only japanese!!! this function split pharse level and save sentence splited.
def wakati(read_path, save_path, mecab_path="/usr/local/lib/mecab/dic/mecab-ipadic-neologd"):
    mecab = MeCab.Tagger(' -Owakati ' + mecab_path)
    with open(read_path) as fs:
        lines = fs.readlines()
    lines = [mecab.parse(line).split("\n")[0] for line in lines]
    print(lines[:10])
    with open(save_path, "w") as fs:
        fs.write("\n".join(lines))
        
##this function return vocabulary set(vocabulary dictionary) from wakatied corpus.
def mk_dict_from_wakatied(read_path):
    with open(read_path) as fs:
        lines = fs.readlines()
    word = []
    lines = [line.split("\n")[0] for line in lines]
    [[word.append(word_) for word_ in line.split(" ")] for line in lines]
    return list(set(word))

##this function save vocabulary set(vocabulary dictionary).
def save_index(file_name, words):
    words = '\n'.join(words)
    with open(file_name, "a") as fs:
        fs.write(words)

##this function read vocabulary set(vocabulary dictionary) from saved.
def read_index(read_path):
    with open(read_path, "r") as fs:
        lines = fs.readlines()
    lines = [line.split("\n")[0] for line in lines]
    return lines
    
## this function marge 2 vocabulary sets.
def marge_vocab(A_vocabs, B_vocabs):
    words = []
    for a_v in A_vocabs:
        words.append(a_v)
    
    for b_v in B_vocabs:
        words.append(b_v)
    return list(set(words))

## Initial input for decoder
def mk_go(batch_size, vocab_size, embedding):
    r = []
    for _ in range(batch_size):
        if embedding:
            r.append(vocab_size)
        else:
            c = [0]*(vocab_size+2)
            c[vocab_size] = 1
            r.append(c)
    return np.reshape(r, (-1, 1)) if embedding else np.array(r)

## this function convert sentence to index which is index of marged vocabrary.
def convert_sentence2index(sentences, index, time_step, go = False):
    r = []
    for sentence in sentences:
        #print(sentence)
        words = sentence.split(" ")
        converted = [index.index(word) for word in words]
        if go:
            converted.insert(0, len(index))
        while len(converted) != time_step and len(converted) <= time_step:
            converted.append(len(index)+1)
        r.append(converted[:time_step])
    return np.reshape(np.array(r), (-1, time_step, 1))

##this function conver sentence to one_hot_encoded  vector ..
def convert_sentence2one_hot_encoding(sentences, indexs, time_step, go=False):
    r = []
    for sentence in sentences:
        words = sentence.split(" ")
        time_steps = []
        ## append <GO>
        if go:
            content = [0]*(len(indexs)+2)
            content[len(indexs)] = 1
            time_steps.append(content)
        
        for word in words:
            content = [0]*(len(indexs)+2)
            idx = indexs.index(word)
            content[idx] = 1
            time_steps.append(content)

        ##append <EOS>
        while len(time_steps) <= time_step and len(time_steps) != time_step:
            content = [0]*(len(indexs)+2)
            content[len(indexs)+1] = 1
            time_steps.append(content)

        r.append(time_steps[:time_step])
    return np.array(r)

## this function return fuction that yields training data for each time steps. And this function used for pre_training.
def mk_training_func(A_wakatied_path, B_wakatied_path, Marged_vocabs_path, batch_size, time_step, embedding=True):
    ## loading reshaped data.. reshaped meas that sentence is splited white space.
    A_corpus = read_corpus(A_wakatied_path)
    B_corpus = read_corpus(B_wakatied_path)
    
    ## Loading each vocabulary and marge them
    indexs = read_index(Marged_vocabs_path)
    
    if embedding:
        convert_func = convert_sentence2index
    else:
        convert_func = convert_sentence2one_hot_encoding
    
    def pre_training_func():
        while True:
            A_choiced_idx = [random.choice(A_corpus) for _ in range(batch_size)]
            B_choiced_idx = [random.choice(B_corpus) for _ in range(batch_size)]
        
            A_in = convert_func(A_choiced_idx, indexs, time_step)
            A_d_in = convert_func(A_choiced_idx, indexs, time_step, True)
            A_d_label = convert_sentence2one_hot_encoding(A_choiced_idx, indexs, time_step)[:,:,:]
            
            B_in = convert_func(B_choiced_idx, indexs, time_step)
            B_d = convert_func(B_choiced_idx, indexs, time_step, True)
            B_d_label = convert_sentence2one_hot_encoding(B_choiced_idx, indexs, time_step)[:,:,:] 
            yield A_in, A_d_in, A_d_label, B_in, B_d, B_d_label
        
    def training_func():
        while True:
            A_choiced_idx = [random.choice(A_corpus) for _ in range(batch_size)]
            B_choiced_idx = [random.choice(B_corpus) for _ in range(batch_size)]
            
            A_in = convert_func(A_choiced_idx, indexs, time_step)
            B_in = convert_func(B_choiced_idx, indexs, time_step)
            yield A_in, B_in
    return pre_training_func, training_func
