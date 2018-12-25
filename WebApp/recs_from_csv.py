import argparse
import csv
import numpy as np
from scipy import sparse


def load_id_mapper():
    mapper_id = {}
    filename = 'static/data/books.csv'
    with open(filename, "r", encoding='utf8') as f:
        reader = csv.DictReader(f, delimiter=",")
        for i, line in enumerate(reader):
            mapper_id[line['goodreads_book_id']] = int(line['book_id'])
    return mapper_id


def create_user_vector(filename, mapper):
    user_vec = np.zeros(10000, dtype=np.float)
    total_valid_reviews = 0
    total_invalid_reviews = 0
    with open(filename, 'r', encoding='utf8') as fp:
        reader = csv.DictReader(fp, delimiter=',')
        for i, line in enumerate(reader):
            goodreads_book_id = line['Book Id']
            if goodreads_book_id in mapper:
                book_id = mapper[goodreads_book_id]
                rating = float(line['My Rating'])
                user_vec[book_id - 1] = rating
                total_valid_reviews += 1
            else:
                print("Couldn't convert {0}".format(line['Title']))
                total_invalid_reviews += 1
    print('Converted user with {0} valid and {1} invalid reviews'.format(total_valid_reviews, total_invalid_reviews))
    return user_vec


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gets recommendations from an exported library")
    parser.add_argument('library_file', help="the path to the library file")
    parser.add_argument('-u', '--username', help="the username to create the file for")
    args = parser.parse_args()

    mapper_id = load_id_mapper()
    user_vec = create_user_vector(args.library_file, mapper_id)
    user_vec = sparse.csr_matrix(user_vec)
    sparse.save_npz('static/data/cached_users/user_{}.npz'.format(args.username), user_vec)
