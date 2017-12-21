#!/usr/bin/env python
# @author = 53 68 79 61 6D 61 6C 
# date	  = 19/12/2017

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import zipfile
from collections import Counter
import random
import utils

FILE_PATH = "./data/text8.zip"

def read_data(filepath):
    """ Reads data from the zip file at filepath"""
    with zipfile.ZipFile(filepath) as f:
        words = tf.compat.as_str(f.read(f.namelist()[0])).split()
         # tf.compat.as_str() converts the input into the string
    return words

def build_vocab(words, vocab_size):
    """builds vocabulary of vocab_size from the words."""
    vocab = dict()
    count = [('UNK', -1)]
    count.extend(Counter(words).most_common(vocab_size - 1))
    index = 0
    utils.make_dir('processed')
    with open('processed/vocab_1000.tsv', "w") as f:
        for word, _ in count:
            vocab[word] = index
            if index < 1000:
                f.write(word + "\n")
            index += 1
    return vocab

def convert_words_to_vocab(words, vocab):
    """Replace all the words in dataset with the index in the vocab"""
    return [vocab[word] if word in vocab else 0 for word in words]

def generate_sample(index_vocab, context_window_size):
    """Generate training pairs according to skip-gram model"""
    for index, center in enumerate(index_vocab):
        context = random.randint(1, context_window_size)
        for target in index_vocab[max(0, index - context) : index]:
            yield center, target
        for target in index_vocab[index + 1 : index + context + 1]:
            yield center, target

def get_batch(iterator, batch_size):
    """generate batches and return them as numpy arrays"""
    while True:
        center_batch = np.zeros([batch_size], dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for ind in range(batch_size):
            center_batch[ind], target_batch[ind] = next(iterator)
        yield center_batch, target_batch

def process_data(vocab_size, context_window_size, batch_size):
    words = read_data(FILE_PATH)
    vocab = build_vocab(words, vocab_size)
    index_words = convert_words_to_vocab(words, vocab)
    del words
    iterator = generate_sample(index_words, context_window_size)
    return get_batch(iterator, batch_size)