import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
from math import ceil
from sklearn.metrics.pairwise import cosine_similarity

# Custom libraries
sys.path.append('../../Util')
from loader import get_book_dataframe, get_book_features
from cross_validation import ColumnwiseKFold
from reduction import get_sparse, reduce_matrix
from joiner import get_ratings, get_reduced_joint
from pipeline import rmse, mae, evaluate, print_evaluation

learning_rate = 1e-3

BASE_DIR = ".tmp"
MODELS_DIR = "{0}/models".format(BASE_DIR)
SUMMARY_DIR = "{0}/summaries".format(BASE_DIR)


class BookEncoder:
    def __init__(self, input_dim, n_hidden=200):
        self.n_hidden = n_hidden
        with tf.variable_scope('inputs'):
            self.items = tf.placeholder(tf.float32, [None, input_dim])

        with tf.variable_scope('encoder'):
            enc1 = tf.layers.dense(self.items, 200, activation=tf.nn.relu)
            self.enc2 = tf.layers.dense(enc1, n_hidden, activation=tf.nn.relu)

        with tf.variable_scope('decoder'):
            dec1 = tf.layers.dense(self.enc2, 200, activation=tf.nn.relu)
            dec2 = tf.layers.dense(dec1, input_dim, activation=tf.nn.relu)

        with tf.variable_scope('loss'):
            self.reconstruction_loss = tf.reduce_mean(tf.pow(self.items - dec2, 2))
            reconstruction_loss_summ = tf.summary.scalar('reconstruction_loss', self.reconstruction_loss)
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.reconstruction_loss)

        with tf.variable_scope('summaries'):
            self.train_summary = tf.summary.merge([reconstruction_loss_summ])

    def initialize(self, session):
        initialize_directories()
        self.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        session.run(tf.global_variables_initializer())
        self.train_writer = tf.summary.FileWriter('{0}/train'.format(SUMMARY_DIR), session.graph)

    def _next_minibatch(self, items, step):
        return items[step * self.batch_size: min((step + 1) * self.batch_size, self.n_train)]

    def encode(self, session, items):
        """ Encodes a set of items """
        [reprs] = session.run([self.enc2], feed_dict={self.items: items})
        return reprs

    def train(self, session, items, max_steps=50000, batch_size=128, print_interval=500):
        self.n_train = items.shape[0]
        self.batch_size = batch_size
        n_batches = ceil(self.n_train * 1.0 / batch_size)
        for i in range(max_steps):
            feed_dict = {
                self.items: self._next_minibatch(items, i % n_batches)
            }
            _, l = session.run([self.optimizer, self.reconstruction_loss], feed_dict=feed_dict)
            if i % 10 == 9:
                [summary] = session.run([self.train_summary], feed_dict=feed_dict)
                self.train_writer.add_summary(summary, i)
            if i % print_interval == print_interval - 1:
                print('Loss: {0:4.7f}'.format(l))

    def test(self, session, original_ratings, held_out_ratings, item_vecs, user_indices, item_indices):
        reprs = self.encode(session, item_vecs)
        sim = (cosine_similarity(reprs) + 1) / 2
        return evaluate(original_ratings, held_out_ratings, sim, user_indices, item_indices)


def initialize_directories():
    dirs = [BASE_DIR, MODELS_DIR, SUMMARY_DIR]
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


def main(ratings_components=100, features_components=100, print_scores=False):
    np.random.seed(42)
    tf.set_random_seed(1984)
    data_path = '../data/goodbooks-10k/'
    # data_path = '../../goodbooks-10k/'
    book_features = get_book_features(get_book_dataframe(data_path))
    reduced_item_features, _, _ = reduce_matrix(book_features, n_components=features_components)

    goodreads_path = '../data/goodbooks-10k/ratings.csv'
    amazon_path = '../data/amazon/ratings_amazon.csv'
    spr = get_ratings(goodreads_path, amazon_path, min_amazon_items=6)

    n_folds = 5
    scores = np.zeros((n_folds, 2))
    kf = ColumnwiseKFold(n_folds, random_seed=30)
    for i, (X, (user_indices, item_indices)) in enumerate(kf.split(spr)):
        _, _, rating_VT = reduce_matrix(X, n_components=ratings_components)
        reduced_item_ratings = rating_VT.T
        items = get_reduced_joint(reduced_item_ratings, reduced_item_features)
        tf.reset_default_graph()
        encoder = BookEncoder(input_dim=items.shape[1], n_hidden=150)
        with tf.Session() as sess:
            encoder.initialize(sess)
            encoder.train(sess, items)
            scores[i, :] = encoder.test(sess, spr, X, items, user_indices, item_indices)
            if print_scores:
                print_evaluation(scores[i, 0], scores[i, 1])

    scores = np.mean(scores, axis=0)
    if print_scores:
        print('{0:d}-Fold Scores:'.format(n_folds))
        print_evaluation(scores[0], scores[1])

    return scores


if __name__ == '__main__':
    main(print_scores=True)
