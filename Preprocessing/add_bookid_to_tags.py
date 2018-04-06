import csv

goodreadsid_to_bookid = {}
filename = 'goodbooks-10k/books.csv'
with open(filename, "r", encoding='utf8') as f:
    reader = csv.reader(f, delimiter=",")
    for i, line in enumerate(reader):
        book_id = line[0]
        goodreads_id = line[1]
        goodreadsid_to_bookid[goodreads_id] = book_id

filename = 'goodbooks-10k/book_tags.csv'
with open('goodbooks-10k/book_tags_with_bookid.csv', 'w') as out_file:
    out_file.write('book_id,goodreads_book_id,tag_id,count\n')

    with open(filename, "r") as in_file:
        reader = csv.reader(in_file, delimiter=",")
        for i, line in enumerate(reader):
            if i > 0: #skip orig header
                book_id = goodreadsid_to_bookid[line[0]]
                new_row = book_id + ',' +  ','.join(line)
                out_file.write(new_row + '\n')