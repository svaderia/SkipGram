#!/usr/bin/env python
# @author = 53 68 79 61 6D 61 6C 
# date	  = 19/12/2017

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import functools

from processing import process_data
import utils


VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128 # Dimention of word embedding vector
SKIP_WINDOW = 1 # The context window
NUM_SAMPLED = 64 # Number of negative examples to sample
LEARNING_RATE = 1
NUM_TRAINING_STEP = 100000
SKIP_STEP = 2000


class SkipGramModel:
    """ Build the graph for word2vec model """
    def __init__(self, vocab_size, batch_size, embed_size, num_sampled, learning_rate):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.lr = learning_rate
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    
    def _data(self):
        self.center_words = tf.placeholder(tf.int32, shape=[self.batch_size], name="center_words")
        self.target_words = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name="taget_words")

    def _create_embedding_matrix(self):
        with tf.name_scope("embeddings"):
            self.embedding_matrix = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0), name="embedding_matrix")
    
    def _create_loss(self):
        with tf.name_scope("loss"):
            embed = tf.nn.embedding_lookup(self.embedding_matrix, self.center_words, name='embed')

            nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size],
                                                        stddev=1.0 / (self.embed_size ** 0.5)), 
                                                        name='nce_weight')
            nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')

            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, 
                                                        biases=nce_bias, 
                                                        labels=self.target_words, 
                                                        inputs=embed, 
                                                        num_sampled=self.num_sampled, 
                                                        num_classes=self.vocab_size), name='loss')
    
    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            opt = tf.train.GradientDescentOptimizer(self.lr)
            self.optimizer = opt.minimize(self.loss, global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        """ Build the graph for our model """
        self._data()
        self._create_embedding_matrix()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()

def train_model(model, batch_gen, num_train_steps):
    saver = tf.train.Saver() # for saving variables like nce_weight, nce_bias, embedding_matrix

    initial_step = 0
    utils.make_dir("checkpoints")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        total_loss = 0.0 # we use this to calculate late average loss in the last SKIP_STEP steps
        writer = tf.summary.FileWriter('improved_graph/lr' + str(LEARNING_RATE), sess.graph)
        initial_step = model.global_step.eval()
        for index in range(initial_step, initial_step + num_train_steps):
            centers, targets = next(batch_gen)
            feed_dict={model.center_words: centers, model.target_words: targets}
            loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op], 
                                              feed_dict=feed_dict)
            writer.add_summary(summary, global_step=index)
            total_loss += loss_batch
            if (index + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                total_loss = 0.0
                saver.save(sess, 'checkpoints/skip-gram', index)

        # code to visualize the embeddings.
        # run "tensorboard --logdir='processed'" to see the embeddings
        final_embed_matrix = sess.run(model.embedding_matrix)
        
        # it has to variable. constants don't work here. you can't reuse model.embed_matrix
        embedding_var = tf.Variable(final_embed_matrix[:1000], name='embedding')
        sess.run(embedding_var.initializer)

        config = projector.ProjectorConfig()
        summary_writer = tf.summary.FileWriter('processed')

        # add embedding to the config file
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        
        # link this tensor to its metadata file, in this case the first 500 words of vocab
        embedding.metadata_path = 'vocab_1000.tsv'

        # saves a configuration file that TensorBoard will read during startup.
        projector.visualize_embeddings(summary_writer, config)
        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(sess, 'processed/model3.ckpt', 1)


def main():
    model = SkipGramModel(VOCAB_SIZE, EMBED_SIZE, BATCH_SIZE, NUM_SAMPLED, LEARNING_RATE)
    model.build_graph()
    batch_gen = process_data(VOCAB_SIZE, SKIP_WINDOW, BATCH_SIZE)
    train_model(model, batch_gen, NUM_TRAINING_STEP)

if __name__ == '__main__':
    main()
