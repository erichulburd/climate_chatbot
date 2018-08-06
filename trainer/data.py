EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz ' # space is included in whitelist
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''
UNK = 'unk'

import random
import argparse

import tensorlayer as tl
import tensorflow as tf
import nltk
import itertools
from collections import defaultdict

from trainer import storage
from io import StringIO
import numpy as np
from tensorflow import gfile
from tensorflow.estimator import ModeKeys
import pickle
import csv
import string

NET_OUT_FILENAME = 'net_out.npy'
NET_RNN_FILENAME = 'net_rnn.npy'

def get_climate_change_questions_and_answers(config):
    questions = []
    answer_tokens = []
    answer_metatokens = {}
    csvs = storage.get('data/climate_augmented_dataset.csv').decode('utf-8')
    augmented_rows = csv.reader(StringIO(csvs), delimiter=',')
    for row in augmented_rows:
        safe_qs = [q.replace('\n', ' ').replace('\r', '') for q in row[0:9]]
        safe_as = row[9].replace('\n', ' ').replace('\r', '')

        # tokenize output as a random string.
        output_length = random.randrange(15, 20, 1)
        output_metatoken = ''.join(random.SystemRandom().choice(string.ascii_uppercase) for _ in range(output_length))
        answer_metatokens[output_metatoken] = safe_as

        for question in safe_qs:
            questions.append(question)
            answer_tokens.append([output_metatoken])
    # This also truncates climate questions that are beyond config limit.
    questions = [q[:config['limit']['maxq']] for q in process_raw_lines(questions)]
    return questions, answer_tokens, answer_metatokens

'''
    1. Read from 'movie-lines.txt'
    2. Create a dictionary with ( key = line_id, value = text )
'''
def get_id2line():
    lines = storage.get('data/movie_lines.txt').decode('utf-8', 'ignore').split('\n')
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            i = _line[0]
            l = _line[4]
            id2line[i] = l
    return id2line

'''
    1. Read from 'movie_conversations.txt'
    2. Create a list of [list of line_id's]
'''
def get_conversations():
    conv_lines = storage.get('data/movie_conversations.txt').decode('utf-8', 'ignore').split('\n')
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
    print(str(filtered) + '%% filtered from original data')

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

def process_data(config, data_directory):
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

    print('\n >> Save numpy arrays to disk')
    # save them
    storage.write(np.ndarray.dumps(idx_q), '%s/idx_q.npy' % data_directory, content_type='text/plain')
    storage.write(np.ndarray.dumps(idx_a), '%s/idx_a.npy' % data_directory, content_type='text/plain')

    # let us now save the necessary dictionaries
    metadata = {
        'w2idx' : w2idx,
        'idx2w' : idx2w,
        'config' : config,
        'freq_dist' : freq_dist,
        'climate_metatokens' : answer_metatokens
    }

    # write to disk : data control dictionaries
    storage.write(pickle.dumps(metadata), '%s/metadata.pkl' % data_directory, content_type='text/plain')
    storage.write(json.dumps(config, indent=2, sort_keys=True), "%s/config.json" % data_directory)

    # count of unknowns
    unk_count = (idx_q == 1).sum() + (idx_a == 1).sum()
    # count of words
    word_count = (idx_q > 1).sum() + (idx_a > 1).sum()

    print('% unknown : {0}'.format(100 * (float(unk_count)/float(word_count))))
    print('Dataset count : ' + str(idx_q.shape[0]))


import numpy as np
import json
from random import sample
import os

'''
 split data into train (90%), test (0%) and valid(10%)
    return tuple( (trainX, trainY), (testX,testY), (validX,validY) )

'''
split_ratios = [0.9, 0, 0.1]
def split_dataset(x, y, ratio = split_ratios):
    # number of examples
    data_len = len(x)
    lens = [ int(data_len*item) for item in ratio ]

    trainX, trainY = x[:lens[0]], y[:lens[0]]
    testX, testY = x[lens[0]:lens[0]+lens[1]], y[lens[0]:lens[0]+lens[1]]
    validX, validY = x[-lens[-1]:], y[-lens[-1]:]

    return (trainX,trainY), (testX,testY), (validX,validY)


'''
 a generic decode function
    inputs : sequence, lookup
'''
def decode(sequence, lookup, separator=''): # 0 used for padding, is ignored
    return separator.join([ lookup[element] for element in sequence if element ])

def length(data_directory, mode):
    idx_q = np.loads(storage.get('%s/idx_q.npy' % data_directory))
    if mode == ModeKeys.TRAIN:
        return len(idx_q) * split_ratios[0]
    elif mode == ModeKeys.EVAL:
        return len(idx_q) * split_ratios[2]
    elif mode == ModeKeys.PREDICT:
        return len(idx_q) * split_ratios[1]

def load_metadata(hypes):
    data_directory = 'working_dir/data/%s' % hypes['data_directory']
    return pickle.loads(
        storage.get("%s/metadata.pkl" % data_directory)
    )

def load_hypes(job_directory):
    return json.loads(
        storage.get("%s/hypes.json" % job_directory).decode('utf-8')
    )

def load_data(hypes):
    data_directory = 'working_dir/data/%s' % hypes['data_directory']

    # read data control dictionaries
    metadata = load_metadata(hypes)
    # read numpy arrays
    idx_q = np.loads(
        storage.get('%s/idx_q.npy' % data_directory)
    )
    idx_a = np.loads(
        storage.get('%s/idx_a.npy' % data_directory)
    )

    (trainX, trainY), (testX, testY), (validX, validY) = split_dataset(idx_q, idx_a)
    trainX = trainX.tolist()
    trainY = trainY.tolist()
    testX = testX.tolist()
    testY = testY.tolist()
    validX = validX.tolist()
    validY = validY.tolist()

    trainX = tl.prepro.remove_pad_sequences(trainX)
    trainY = tl.prepro.remove_pad_sequences(trainY)
    validX = tl.prepro.remove_pad_sequences(validX)
    validY = tl.prepro.remove_pad_sequences(validY)
    testX = tl.prepro.remove_pad_sequences(testX)
    testY = tl.prepro.remove_pad_sequences(testY)

    return metadata, trainX, trainY, testX, testY, validX, validY

def get_start_id(metadata):
    return len(metadata['idx2w'])

def get_end_id(metadata):
    return len(metadata['idx2w']) + 1

def preprocess_data(metadata, X, Y, hypes, mode):
    xseq_len = len(X)
    yseq_len = len(Y)
    assert xseq_len == yseq_len
    # n_step = int(xseq_len/batch_size)
    xvocab_size = len(metadata['idx2w']) # 8002 (0~8001)
    # emb_dim = hypes['emb_dim']

    w2idx = metadata['w2idx']   # dict  word 2 index
    idx2w = metadata['idx2w']   # list index 2 word

    # unk_id = w2idx['unk']   # 1
    # pad_id = w2idx['_']     # 0

    start_id = get_start_id(metadata)  # 8002
    end_id = get_end_id(metadata)  # 8003

    w2idx.update({'start_id': start_id})
    w2idx.update({'end_id': end_id})
    idx2w = idx2w + ['start_id', 'end_id']

    xvocab_size = end_id + 1

    X = tl.prepro.pad_sequences(X)
    target_seqs = tl.prepro.sequences_add_end_id(Y, end_id=end_id)
    target_seqs = tl.prepro.pad_sequences(target_seqs)

    decode_seqs = tl.prepro.sequences_add_start_id(Y, start_id=start_id, remove_last=False)
    decode_seqs = tl.prepro.pad_sequences(decode_seqs)
    target_mask = tl.prepro.sequences_get_mask(target_seqs)

    # for i in range(100):
    #    print(i, [idx2w[id] for id in X[i]])
    #    print(i, [idx2w[id] for id in Y[i]])
    #    print(i, [idx2w[id] for id in target_seqs[i]])
    #    print(i, [idx2w[id] for id in decode_seqs[i]])
    #    print(i, target_mask[i])
    #    print(len(target_seqs[i]), len(decode_seqs[i]), len(target_mask[i]))
    # exit()

    features = {
        'encode_seqs': tf.stack(X),
        'decode_seqs': tf.stack(decode_seqs)
    }

    labels = {
        'target_seqs': tf.stack(target_seqs),
        'target_mask': tf.stack(target_mask)
    }

    return features, labels

