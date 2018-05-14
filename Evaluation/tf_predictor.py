import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
from math import ceil
from sklearn.metrics.pairwise import cosine_similarity

# Custom libraries
sys.path.append('../Util')
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
    def __init__(self, user_input_dim, book_input_dim, user_hidden=200, book_hidden=200):
        self.user_input_dim = user_input_dim
        self.book_input_dim = book_input_dim
        self.user_hidden = user_hidden
        self.book_hidden = book_hidden
        with tf.variable_scope('inputs'):
            self.single_user = tf.placeholder(tf.float32, [1, user_input_dim])
            self.users = tf.placeholder(tf.float32, [None, user_input_dim])
            self.items = tf.placeholder(tf.float32, [None, book_input_dim])
            self.actual_ratings = tf.placeholder(tf.float32, [None, 1])

        with tf.variable_scope('users'):
            self.users_enc_out = self.get_user_encoder(self.users)
            self.users_dec_out = self.get_user_decoder(self.users_enc_out)
            self.get_user_reconstruction_loss()

        with tf.variable_scope('books'):
            self.books_enc_out = self.get_book_encoder(self.items)
            self.books_dec_out = self.get_book_decoder(self.books_enc_out)
            self.get_book_reconstruction_loss()

        with tf.variable_scope('users', reuse=True):
            self.user_enc = self.get_user_encoder(self.single_user)

        # Concatenate user encoding with book encodings
        # First get the number of books
        num_books = tf.shape(self.books_enc_out)[0]

        # Then tile the user encoding among the first axis
        tiled_user_enc = tf.tile(self.user_enc, tf.stack([num_books, 1]))

        # Now we can concatenate user with books
        self.combined = tf.concat([tiled_user_enc, self.books_enc_out], 1)

        # Throw a few dense layers for good measure
        with tf.variable_scope('final_dense'):
            self.combined = tf.layers.dense(self.combined, 150, activation=tf.nn.relu)
            self.pred_ratings = tf.layers.dense(self.combined, 1, activation=None)

        # Get loss
        with tf.variable_scope('rating_loss'):
            self.rating_loss = tf.sqrt(tf.reduce_mean(tf.square(self.pred_ratings - self.actual_ratings)))
            self.rating_optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.rating_loss)

        # with tf.variable_scope('summaries'):
        #     self.train_summary = tf.summary.merge([reconstruction_loss_summ])

    def get_user_encoder(self, encoder_in):
        with tf.variable_scope('user_encoder'):
            enc1 = tf.layers.dense(encoder_in, 300, activation=tf.nn.relu)
            enc2 = tf.layers.dense(enc1, self.user_hidden, activation=tf.nn.relu)
        return enc2

    def get_user_decoder(self, decoder_in):
        with tf.variable_scope('user_decoder'):
            dec1 = tf.layers.dense(decoder_in, 300, activation=tf.nn.relu)
            dec2 = tf.layers.dense(dec1, self.user_input_dim, activation=tf.nn.relu)
        return dec2

    def get_book_encoder(self, encoder_in):
        with tf.variable_scope('book_encoder'):
            enc1 = tf.layers.dense(encoder_in, 300, activation=tf.nn.relu)
            enc2 = tf.layers.dense(enc1, self.book_hidden, activation=tf.nn.relu)
        return enc2

    def get_book_decoder(self, decoder_in):
        with tf.variable_scope('book_decoder'):
            dec1 = tf.layers.dense(decoder_in, 300, activation=tf.nn.relu)
            dec2 = tf.layers.dense(dec1, self.book_input_dim, activation=tf.nn.relu)
        return dec2

    def get_user_reconstruction_loss(self):
        with tf.variable_scope('user_reconstruction_loss'):
            self.user_reconstruction_loss = tf.reduce_mean(tf.pow(self.users - self.users_dec_out, 2))
            self.user_reconstruction_optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.user_reconstruction_loss)

    def get_book_reconstruction_loss(self):
        with tf.variable_scope('book_reconstruction_loss'):
            self.book_reconstruction_loss = tf.reduce_mean(tf.pow(self.items - self.books_dec_out, 2))
            # reconstruction_loss_summ = tf.summary.scalar('reconstruction_loss', self.reconstruction_loss)
            self.book_reconstruction_optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.book_reconstruction_loss)

    def initialize(self, session):
        initialize_directories()
        self.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        session.run(tf.global_variables_initializer())
        # self.train_writer = tf.summary.FileWriter('{0}/train'.format(SUMMARY_DIR), session.graph)

    def _next_minibatch(self, users, items, step):
        user_vec = users[step, :].todense()
        item_vec = items[user_vec.nonzero()[1], :]
        return user_vec, item_vec

    def encode(self, session, items):
        """ Encodes a set of items """
        [reprs] = session.run([self.enc2], feed_dict={self.items: items})
        return reprs

    def train(self, session, users, items, max_steps=50000, batch_size=1, print_interval=500):
        self.n_train = users.shape[0]
        self.batch_size = batch_size
        n_batches = ceil(self.n_train * 1.0 / batch_size)
        for i in range(max_steps):
            user_vec, item_vec = self._next_minibatch(users, items, i % n_batches)
            feed_dict = {
                self.single_user: user_vec,
                self.items: item_vec,
                self.actual_ratings: user_vec[user_vec.nonzero()].reshape((-1, 1)),
            }
            _, l = session.run([self.rating_optimizer, self.rating_loss], feed_dict=feed_dict)
            # if i % 10 == 9:
            #     [summary] = session.run([self.train_summary], feed_dict=feed_dict)
            #     self.train_writer.add_summary(summary, i)
            if i % print_interval == print_interval - 1:
                print('Loss: {0:4.7f}'.format(l))

    def test(self, session, original_ratings, held_out_ratings, item_vecs, user_indices, item_indices):
        held_outs = original_ratings[(user_indices, item_indices)]
        predictions = []
        for u, i in zip(user_indices, item_indices):
            user_arr = np.asarray(held_out_ratings[u,:].todense())[0]
            item_arr = item_vecs[i,:]
            feed_dict = {
                self.single_user: user_arr.reshape((1,-1)),
                self.items: item_arr.reshape((-1,item_arr.shape[0])),
            }
            pred_rating = session.run([self.pred_ratings], feed_dict=feed_dict)
            predictions.append(pred_rating[0][0][0])
        print(len(predictions))
        print(predictions[0])
        predictions = np.array(predictions)
        rmse_ = rmse(held_outs, predictions)
        mae_ = mae(held_outs, predictions)
        return rmse_, mae_


def initialize_directories():
    dirs = [BASE_DIR, MODELS_DIR, SUMMARY_DIR]
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


def main(ratings_components=300, features_components=300, print_scores=False):
    np.random.seed(42)
    tf.set_random_seed(1984)
    data_path = '../data/goodbooks-10k/'
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
        encoder = BookEncoder(user_input_dim=10000, book_input_dim=items.shape[1], user_hidden=150, book_hidden=150)
        with tf.Session() as sess:
            encoder.initialize(sess)
            encoder.train(sess, X, items)
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
