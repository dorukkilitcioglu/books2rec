import ast

def main():
	seen = 0

	file = open('mapper_asin_to_name.csv', 'w')
	with open('amazon_data/meta_Books.json', 'r') as fp:
		for line in fp:
			if seen % 100000 == 0:
				print(seen)
			a = ast.literal_eval(line)
			if 'title' in a:
				file.write(a['asin'] + ',' + a['title'] + '\n')
			else:
				print("No title")
				
			seen += 1
	file.close()
main()