import ast
import re
import html
import io

dirty = [
'hardcover',
'paperback',
'volume',
'unabridged',
'boxed Set',
'collection',
'audiobook',
'publication'
]

def main():
	seen = 0

	file = open('mapper_asin_to_name_clean.csv', 'w')
	with open('mapper_asin_to_name.csv', 'r') as fp:
		for line in fp:
			line = line.lower()
			orig = line
			
			# clean html
			line = html.unescape(line)
			
			# remove dirty words
			for word in dirty:
				line = line.replace(word, '') 
				
			# remove all () and []
			line = re.sub("[\(\[].*?[\)\]]", "", line)
				
			try:
				file.write(line)
			except:
				print("ERROR WITH: " + line)
	file.close()
main()