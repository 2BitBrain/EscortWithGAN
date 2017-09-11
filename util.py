import os
from urllib.request import urlopen
import re
import numpy as np

def mk_train_and_test_data(save_dir_path):
    if not os.path.exists(save_dir_path):
        os.mkdir(save_dir_path)

    def read_content_save(url, save_dir_path, name):
        content = urlopen(url).read()
        with open(save_dir_path+name, "w") as fs:
            fs.write(content)
        print("success")

    train_data_url = "https://storage.googleapis.com/kaggle-competitions-data/inclass/255etraining.txt?GoogleAccessId=competitions-data@kaggle-161607.iam.gserviceaccount.com&Expires=1502423006&Signature=lBHL5Z3NRdNyAUa%2FaKeWLn9alHhpDW3v4uxlfSuPMojTyiMSwmfVRthsupDnnsSmY1mLyCtUbSUpf%2FxH6nwdBlRgeCwpXVhLLKNDRzc%2B8ZRnWebaK%2FTHVj23Pp%2FON2f5QaEyApdeULrz2RnT07fl8gnTXcMQKug8CcnvKiJAAHCHKoNjV8T9q%2F8S5sYq7wgCuX9C3tLVlIxSP1ozQF2pVDbObCkzODEiF5pVX5Rp8yRHWgCOqoSNvAUKBp0EE8DoE7ncbTCBnFl7%2FFpjHi5QGya582A%2BRm1jaAzaMDrqVoKF4Y8llmIXt%2FyWOn5fUoZpd1uYp%2F8TURXt2hxCY1gxNQ%3D%3D"
    test_data_url = "https://storage.googleapis.com/kaggle-competitions-data/inclass/2558/testdata.txt?GoogleAccessId=competitions-data@kaggle-161607.iam.gserviceaccount.com&Expires=1502422338&Signature=WD%2BoEzW8x%2BnGFl1nB0NCT8Ru5fV%2FtGcv9Gqp6188da6IFbSrKgOly%2FfBKzW%2FKl%2BZiF%2F76R2o%2BeUaWbQKtlf37nOr%2FgXjJMKEfua9UUA%2BfbqZ5P4ItIutUnoEWswFZYY31kSwGTD1ex7kof7nse6zWi2B2shJND8tKK1hyzRBwSnOtl8qQSi1LKGGJVL9XBc4ZlpHnH8m54bO985JWCt85yHNScxY%2BzbDoi2XszVHfHG%2Bue4RN82uXcTvycrLKTKnmELArbDMcZaQsJ7vhrfy2XQRxL8yJsrHmTlxKcGvU11GTAkCrfyNsbjiMVkefaSo1NtB2eDBATZICOKnMHFObQ%3D%3D"
    read_content_save(train_data_url, save_dir_path, "train.txt")
    read_content_save(test_data_url, save_dir_path, "test.txt")
    
def save_index(unique_word_list, save_path):
    #assert not type(unique_word_list) == "list" , "unique_word_list type is not list.please cast it"

    content = "\n".join(list(map(str, unique_word_list)))
    with open(save_path, "w") as fs:
        fs.write(content)

    print("success")

def read_index(data_path):
    with open(data_path, "r") as fs:
        lines = fs.readlines()
    
    return [line.split("\n")[0] for line in lines]
     
    
def remove_anti_pattern(sentence, patterns=[["\.+", " ."], ["\!+", " !"], ["\?+", " ?"], [",", " ,"], ["[\n\x99\x92\x80@�£ã’‘©µ…ªâ*&\]\“^[\-><_;:+#”$%'\"()=~|¥{}/\\\\]", ""],["\s{2,}", ""],["\.", " ."]]):
        for pattern in patterns:
            sentence = re.sub(pattern[0], pattern[1], sentence)
        return sentence    

def read_training_data(data_path):
    with open(data_path, "r") as fs:
        lines = fs.readlines()
   
    pos = []
    neg = []

    label = [int(line.split("	")[0]) for line in lines]
    sentences = [remove_anti_pattern(line.split("	")[1].lower()) for line in lines]
    for label_, sentence in zip(label, sentences):
        if label_ == 0:
            neg.append(sentence)
        else:
            pos.append(sentence)

    return neg, pos 

def mk_go(batch_size, vocab_size):
    r = []
    for _ in range(batch_size):
        c = [0]*vocab_size
        c[vocab_size-1] = 1
        r.append(c)
    return np.array(r)

def convert_sentence2index(sentences, index, time_step):
    r = []
    for sentence in sentences:
        #print(sentence)
        words = sentence.split(" ")
        converted = [index.index(word) for word in words]
        while len(converted) != time_step and len(converted) <= time_step:
            converted.append(len(index))
        r.append(converted[:time_step])
    return np.reshape(np.array(r), (-1, time_step, 1))


def convert_label(labels):
    r = []
    for label in labels:
        content = [0]*2
        content[int(label)] = 1
        r.append(content)
    return np.array(r)

def convert_senteo2one_hot_encoding(sentences, indexs, time_step):
    r = []
    for sentence in sentences:
        words = sentence.split(" ")
        time_steps = []
        for word in words:
            content = [0]*(len(indexs)+2)
            idx = indexs.index(word)
            content[idx] = 1
            time_steps.append(content)

        while len(time_steps) <= time_step and len(time_steps) != time_step:
            content = [0]*(len(indexs)+2)
            content[len(indexs)] = 1
            time_steps.insert(0, content)

        r.append(time_steps[:time_step])
    return np.array(r)

def visualizer(x, y, index_path, save_path):
    indexs = read_index(index_path)
    sentences_x = []
    sentences_y = []
    for s_x, s_y in zip(x, y):
        idxs_x = np.argmax(s_x, axis=-1)
        idxs_y = np.argmax(s_y, axis=-1)
        sentences_x.append(" ".join([indexs[idx] for idx in idxs_x if len(indexs) > idx]))
        sentences_y.append(" ".join([indexs[idx] for idx in idxs_y if len(indexs) > idx]))


    with open(save_path, "a") as fs:
        fs.write("\n".join(["{}  <-> {}".format(x,y) for x,y in zip(sentences_x, sentences_y)]))

"""
def mk_train_data(data_path, index_path, time_step):
    neg_sentences, pos_sentences = read_training_data(data_path)
    print(len(neg_sentences), len(pos_sentences))
    if  not os.path.exists(index_path):
        word = []
        for r_text in sentences:
            #print(r_text)
            [word.append(word_) for word_ in r_text.split(' ')]
        save_index(set(word), index_path)
    
    indexs = read_index(index_path)
    neg_converted_sentences = convert_sentence2index(neg_sentences, indexs, time_step)
    print("Done converting negative sentences")
    neg_one_hot_sentences = convert_senteo2one_hot_encoding(neg_sentences, indexs, time_step)
    print("Done converting negative sentences to one hot vector")
    pos_converted_sentences = convert_sentence2index(pos_sentences, indexs, time_step)
    print("Done converting postive sentences")
    pos_one_hot_sentences = convert_senteo2one_hot_encoding(pos_sentences, indexs, time_step)
    print("Done converting positive sentences to one hot vector")
    return neg_converted_sentences, neg_one_hot_sentences, pos_converted_sentences, pos_one_hot_sentences
"""

def mk_train_data(data_path, index_path, time_step,embedding=False):
    neg_sentences, pos_sentences = read_training_data(data_path)
    sentences = read_sentence_data(data_path)
    if  not os.path.exists(index_path):
        word = []
        for r_text in sentences:
            [word.append(word_) for word_ in r_text.split(' ')]
        save_index(set(word), index_path)
        
    indexs = read_index(index_path)
    
    if embedding:
        convert_func = convert_sentence2index
    else:
        convert_func = convert_sentence2one_hot_encoding
        
    def mk_pre_train_func(embedding=False):
            in_neg = convert_func(neg_sentences, indexs, time_step)
            d_in_neg = convert_func(neg_sentences, indexs, time_step, True)
            d_label_neg = convert_sentence2one_hot_encoding(neg_sentences, indexs, time_step)[:,:,:]
            
            in_pos = convert_func(pos_sentences, indexs, time_step)
            d_pos = convert_func(pos_sentences, indexs, time_step, True)
            d_label_pos = convert_sentence2one_hot_encoding(pos_sentences, indexs, time_step)[:,:,:] 
            return in_neg, d_in_neg, d_label_neg, in_pos, d_pos, d_label_pos

    def mk_train_func(embedding=False):
        in_neg = convert_func(neg_sentences, indexs, time_step)
        in_pos = convert_func(pos_sentences, indexs, time_step)
        return in_neg, in_pos
    
    return mk_pre_train_func, mk_train_func