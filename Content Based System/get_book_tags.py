import csv

bad_tags = {
    'to-read',
    'currently-reading',
    'books-i-own',
    'owned',
    'owned-books',
    'read',
    'favourites',
    'default',
    'kindle',
    'my-books',
    'to-buy',
    'all-time-favorites',
    're-read',
    'i-own',
    'ebook',
    'on-hold',
    'favorite',
    'favorites'
}

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
            if tag_name not in bad_tags:
                books[goodreads_book_id][tag_name] = count
    return books