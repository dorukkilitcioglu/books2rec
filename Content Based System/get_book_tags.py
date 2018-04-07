import csv
from util import bad_features

def get_tags(book_tags, tags):
    tag_defs = {}
    with open(tags, "r", encoding='utf8') as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            tag_defs[line[0]] = line[1]

    books = {}
    with open(book_tags, "r", encoding='utf8') as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            goodreads_book_id = line[1]
            tag_id = line[2]
            count = line[3]
            if goodreads_book_id not in books:
                books[goodreads_book_id] = {}
            
            tag_name = tag_defs[tag_id]
            if tag_name not in bad_features:
                books[goodreads_book_id][tag_name] = count
    return books