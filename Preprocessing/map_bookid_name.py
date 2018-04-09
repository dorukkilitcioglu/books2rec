import ast
import csv
import pandas as pd 

import sys

maxInt = sys.maxsize
decrement = True

while decrement:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True

def clean(s):
	s = s.lower().strip()
	s = s.replace('&', 'and')
	s = ''.join([i for i in s if (i.isalpha() or i.isspace())])
	s = ' '.join(s.split())
	return s

def main():
	with open('goodbooks-10k/books.csv', 'r', encoding="utf-8") as f:
		reader = csv.reader(f)
		books = list(reader)
		
	mapper = {}
	mapper_original = {}
	for book in books:
		book_id = book[0]
		
		original_title = clean(book[9])
		
		title = clean(book[10])
		
		if book_id != 'book_id':
			mapper_original[original_title] = book_id
			mapper[title] = book_id
		
	print("Number of books: %s" % len(books))
	
	file = open('ratings_amazon.csv', 'w')
	file.write('user_id,book_id,rating\n')
	
	file_amazon = open('ratings_amazon_not_in_goodreads.csv', 'w')
	file_amazon.write('user_id,book_id,rating\n')
	
	with open('amazon_data/ratings_with_name.csv', 'r', encoding="utf-8") as f:
		reader = csv.reader(f)
		for rating in reader:
			name = clean(rating[4])
			user_id = rating[0]
			val = int(float(rating[2]))
			asin = rating[1]
			
			book_id = None
			if name in mapper:
				book_id = mapper[name]
			if book_id == None and name in mapper_original:
				book_id = mapper_original[name]
			if book_id:
				row = user_id + ',' + book_id + ',' + str(val) + '\n'
				file.write(row)
			else:
				row = user_id + ',' + asin + ',' + str(val) + '\n'
				file_amazon.write(row)
	file.close()
	
main()