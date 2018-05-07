import ast
import csv

def main():

	# create dictionary of asin to name
	mapper = {}
	no_asin = 0
	for line in open("mapper_asin_to_name_clean.csv"):
		split = line.strip().split(',', 1)
		if len(split) == 2:
			asin = split[0]
			name = split[1]
			
			mapper[asin] = name
		else:
			no_asin += 1
	print(no_asin)

	file = open('amazon_data/ratings_with_name.csv', 'w', encoding='utf-8')
	no_asin = 0
	for line in open("amazon_data/ratings_Books.csv"):
		split = line.strip().split(',')
		asin = split[1]
		if asin not in mapper:
			no_asin += 1
		else:
			split.append(mapper[asin])
			myString = ",".join(split)
			file.write(myString + '\n')
	print(no_asin)
		
	file.close()
main()