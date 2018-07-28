EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz ' # space is included in whitelist
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''
UNK = 'unk'

import random

import nltk
import itertools
from collections import defaultdict

import numpy as np
from tensorflow import gfile
import pickle
from datetime import datetime
import csv
import string

def get_climate_change_questions_and_answers(config):
    questions = []
    answer_tokens = []
    answer_metatokens = {}
    f = open('data/climate_augmented_dataset.csv', encoding='utf-8', errors='ignore')
    augmented_rows = csv.reader(f, delimiter=',')
    for row in augmented_rows:
        safe_qs = [q.replace('\n', ' ').replace('\r', '') for q in row[0:9]]
        safe_as = row[9].replace('\n', ' ').replace('\r', '')

        # tokenize output as a random string.
        output_length = random.randrange(15, 20, 1)
        output_metatoken = ''.join(random.SystemRandom().choice(string.ascii_uppercase) for _ in range(output_length))
        answer_metatokens[output_metatoken] = safe_as

        for question in safe_qs:
            questions.append(question)
            answer_tokens.append(output_metatoken)
    # This also truncates climate questions that are beyond config limit.
    questions = [q[:config['limit']['maxq']] for q in process_raw_lines(questions)]
    return questions, answer_tokens, answer_metatokens

'''
    1. Read from 'movie-lines.txt'
    2. Create a dictionary with ( key = line_id, value = text )
'''
def get_id2line():
    lines = open('data/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]
    return id2line

'''
    1. Read from 'movie_conversations.txt'
    2. Create a list of [list of line_id's]
'''
def get_conversations():
    conv_lines = open('data/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')
    convs = [ ]
    for line in conv_lines[:-1]:
        _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
        convs.append(_line.split(','))
    return convs

'''
    Get lists of all conversations as Questions and Answers
    1. [questions]
    2. [answers]
'''
def gather_dataset(convs, id2line):
    questions = []; answers = []

    for conv in convs:
        if len(conv) % 2 != 0:
            conv = conv[:-1]
        for i in range(len(conv)):
            if i % 2 == 0:
                questions.append(id2line[conv[i]])
            else:
                answers.append(id2line[conv[i]])

    return questions, answers


'''
    We need 4 files
    1. train.enc : Encoder input for training
    2. train.dec : Decoder input for training
    3. test.enc  : Encoder input for testing
    4. test.dec  : Decoder input for testing
'''
def prepare_seq2seq_files(questions, answers, path, TESTSET_SIZE = 30000):

    # open files
    train_enc = open(path + 'train.enc','w')
    train_dec = open(path + 'train.dec','w')
    test_enc  = open(path + 'test.enc', 'w')
    test_dec  = open(path + 'test.dec', 'w')

    # choose 30,000 (TESTSET_SIZE) items to put into testset
    test_ids = random.sample([i for i in range(len(questions))],TESTSET_SIZE)

    for i in range(len(questions)):
        if i in test_ids:
            test_enc.write(questions[i]+'\n')
            test_dec.write(answers[i]+ '\n' )
        else:
            train_enc.write(questions[i]+'\n')
            train_dec.write(answers[i]+ '\n' )
        if i%10000 == 0:
            print('\n>> written {} lines'.format(i))

    # close files
    train_enc.close()
    train_dec.close()
    test_enc.close()
    test_dec.close()

'''
 filter too long and too short sequences
    return tuple( filtered_ta, filtered_en )

'''
def filter_data(qseq, aseq, config):
    filtered_q, filtered_a = [], []
    raw_data_len = len(qseq)

    assert len(qseq) == len(aseq)

    limit = config['limit']
    for i in range(raw_data_len):
        qlen, alen = len(qseq[i]), len(aseq[i])
        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= limit['maxa']:
                filtered_q.append(qseq[i])
                filtered_a.append(aseq[i])

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len)*100/raw_data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a


'''
 read list of words, create index to word,
  word to index dictionaries
    return tuple( vocab->(word, count), idx2w, w2idx )

'''
def index_(tokenized_sentences, vocab_size, keep=[]):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = ['_'] + [UNK] + [ x[0] for x in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    replacements = []
    for token in keep:
        if token not in word2index:
            replacements.append(token)
    for index, keep_token in enumerate(replacements):
        replacement_index = len(index2word) - (len(replacements) - index)
        token_to_replace = index2word[replacement_index]
        index2word[replacement_index] = keep_token
        word2index[keep_token] = replacement_index
        del word2index[token_to_replace]

    return index2word, word2index, freq_dist

'''
 filter based on number of unknowns (words not in vocabulary)
  filter out the worst sentences

'''
def filter_by_unk_ratio(qtokenized, atokenized, w2idx, config):
    data_len = len(qtokenized)

    filtered_q, filtered_a = [], []

    for qline, aline in zip(qtokenized, atokenized):
        unk_count_q = sum([ 1 for w in qline if w not in w2idx ])
        unk_count_a = sum([ 1 for w in aline if w not in w2idx ])
        q_ok = (float(unk_count_q)/float(len(qline))) <= config['permissible_unk_ratio']
        a_ok = (float(unk_count_a)/float(len(aline))) <= config['permissible_unk_ratio']
        if a_ok & q_ok:
            filtered_q.append(qline)
            filtered_a.append(aline)

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((data_len - filt_data_len)*100/data_len)
    print(str(filtered) + '% filtered for unknown ratio')

    return filtered_q, filtered_a




'''
 create the final dataset :
  - convert list of items to arrays of indices
  - add zero padding
      return ( [array_en([indices]), array_ta([indices]) )

'''
def vectorize_question_and_answers(qtokenized, atokenized, w2idx, config):
    # num of rows
    data_len = len(qtokenized)

    limit = config['limit']
    # numpy arrays to store indices
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = vectorize_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = vectorize_seq(atokenized[i], w2idx, limit['maxa'])

        #print(len(idx_q[i]), len(q_indices))
        #print(len(idx_a[i]), len(a_indices))
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a


'''
 replace words with indices in a sequence
  replace with unknown if word not in lookup
    return [list of indices]

'''
def vectorize_seq(seq, w2idx, maxlen):
    indices = []
    for word in seq:
        if word in w2idx:
            indices.append(w2idx[word])
        else:
            indices.append(w2idx[UNK])
    return indices + [0]*(maxlen - len(seq))

def process_raw_lines(lines):
    return [line_to_token(line) for line in lines ]

def line_to_token(line):
    return [w.strip() for w in filter_line(line.lower(), EN_WHITELIST).split(' ') if w]

'''
 remove anything that isn't in the vocabulary
    return str(pure en)

'''
def filter_line(line, whitelist):
    return ''.join([ ch for ch in line if ch in whitelist ])

def process_data(config, directory):
    id2line = get_id2line()
    print('>> gathered id2line dictionary.\n')
    convs = get_conversations()
    print(convs[121:125])
    print('>> gathered conversations.\n')
    questions, answers = gather_dataset(convs,id2line)

    # change to lower case (just for en)
    qtokenized = process_raw_lines(questions)
    atokenized = process_raw_lines(answers)

    # filter out too long or too short sequences
    print('\n>> 2nd layer of filtering')
    qtokenized, atokenized = filter_data(qtokenized, atokenized, config)

    print('\n:: Sample from segmented list of words')
    for q,a in zip(qtokenized[141:145], atokenized[141:145]):
        print('q : {0}; a : {1}'.format(q,a))

    climate_qtokenized, climate_atokenized, answer_metatokens = get_climate_change_questions_and_answers(config)
    qtokenized += climate_qtokenized
    atokenized += climate_atokenized

    # indexing -> idx2w, w2idx
    print('\n >> Index words')
    idx2w, w2idx, freq_dist = index_( qtokenized + atokenized, config['vocab_size'], keep=answer_metatokens.keys())

    # filter out sentences with too many unknowns
    print('\n >> Filter Unknowns')
    qtokenized, atokenized = filter_by_unk_ratio(qtokenized, atokenized, w2idx, config)
    print('\n Final dataset len : ' + str(len(qtokenized)))

    print('\n >> Vectorizing inputs')
    idx_q, idx_a = vectorize_question_and_answers(qtokenized, atokenized, w2idx, config)

    os.mkdir(directory)

    print('\n >> Save numpy arrays to disk')
    # save them
    np.save('%s/idx_q.npy' % directory, idx_q)
    np.save('%s/idx_a.npy' % directory, idx_a)

    # let us now save the necessary dictionaries
    metadata = {
        'w2idx' : w2idx,
        'idx2w' : idx2w,
        'config' : config,
        'freq_dist' : freq_dist
    }

    # write to disk : data control dictionaries
    file_path = '%s/metadata.pkl' % directory
    with open(file_path, 'wb') as f:
        pickle.dump(metadata, f)

    # save answer metatokens
    with open("%s/climate_augmented_metatokens.json" % directory, "w") as f:
      json.dump(answer_metatokens, f, indent=2, sort_keys=True)

    # count of unknowns
    unk_count = (idx_q == 1).sum() + (idx_a == 1).sum()
    # count of words
    word_count = (idx_q > 1).sum() + (idx_a > 1).sum()

    print('% unknown : {0}'.format(100 * (float(unk_count)/float(word_count))))
    print('Dataset count : ' + str(idx_q.shape[0]))


    #print '>> gathered questions and answers.\n'
    #prepare_seq2seq_files(questions,answers)


import numpy as np
import json
from random import sample
import os

'''
 split data into train (70%), test (15%) and valid(15%)
    return tuple( (trainX, trainY), (testX,testY), (validX,validY) )

'''
def split_dataset(x, y, ratio = [0.7, 0.15, 0.15] ):
    # number of examples
    data_len = len(x)
    lens = [ int(data_len*item) for item in ratio ]

    trainX, trainY = x[:lens[0]], y[:lens[0]]
    testX, testY = x[lens[0]:lens[0]+lens[1]], y[lens[0]:lens[0]+lens[1]]
    validX, validY = x[-lens[-1]:], y[-lens[-1]:]

    return (trainX,trainY), (testX,testY), (validX,validY)


'''
 generate batches from dataset
    yield (x_gen, y_gen)

    TODO : fix needed

'''
def batch_gen(x, y, batch_size):
    # infinite while
    while True:
        for i in range(0, len(x), batch_size):
            if (i+1)*batch_size < len(x):
                yield x[i : (i+1)*batch_size ].T, y[i : (i+1)*batch_size ].T

'''
 generate batches, by random sampling a bunch of items
    yield (x_gen, y_gen)

'''
def rand_batch_gen(x, y, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield x[sample_idx].T, y[sample_idx].T

#'''
# convert indices of alphabets into a string (word)
#    return str(word)
#
#'''
#def decode_word(alpha_seq, idx2alpha):
#    return ''.join([ idx2alpha[alpha] for alpha in alpha_seq if alpha ])
#
#
#'''
# convert indices of phonemes into list of phonemes (as string)
#    return str(phoneme_list)
#
#'''
#def decode_phonemes(pho_seq, idx2pho):
#    return ' '.join( [ idx2pho[pho] for pho in pho_seq if pho ])


'''
 a generic decode function
    inputs : sequence, lookup
'''
def decode(sequence, lookup, separator=''): # 0 used for padding, is ignored
    return separator.join([ lookup[element] for element in sequence if element ])



def get_config(config_file="data_config.json"):
    return json.load(open(config_file))

if __name__ == '__main__':
    directory = 'working_dir/data/%s' % datetime.now().strftime('%Y_%m_%d_%H.%M')
    process_data(get_config(), directory)

def load_data(timestamp):
    # read data control dictionaries
    file_path = "working_dir/data/%s/metadata.pkl" % timestamp
    with open(file_path, 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_q = np.load('working_dir/data/%s/idx_q.npy' % timestamp)
    idx_a = np.load('working_dir/data/%s/idx_a.npy' % timestamp)
    return metadata, idx_q, idx_a
