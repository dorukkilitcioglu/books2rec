import os
import pandas as pd
import xml_to_dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Function that takes in book title as input and outputs most similar book
def get_recommendations(df, indices, title, cosine_sim):
    # Get the index of the book that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all books with that book
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 30 most similar books
    sim_scores = sim_scores[1:31]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar books
    return df['title'].iloc[book_indices]

#TODO - make this path not hardcoded
metadata_directory = '../../goodbooks-10k/books_xml/books_xml'
print(metadata_directory)

books = []
for file in os.listdir(metadata_directory):
    filename = metadata_directory + '/' + os.fsdecode(file)
    raw_book, popular_shelves = xml_to_dict.dict_from_xml_file(filename)

    book = {}
    book['id'] = raw_book['book']['id']
    book['title'] = raw_book['book']['title']

    book['description'] = raw_book['book']['description']
    # TODO - clean description

    book['popular_shelves'] = popular_shelves
    # Turn popular shelves into soup
    soup = ''
    normalizing_value = 5
    for key,value in popular_shelves.items():
        for i in range(int(value) // normalizing_value):
            soup += ' ' + key
    
    if book['description']:
        book['description'] = book['description'] + soup
    else:
        book['description'] = soup

    books.append(book)

df = pd.DataFrame(books)
# df = df.set_index('id')

#Replace NaN with an empty string
df['description'] = df['description'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(df['description'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and book titles
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

title = 'The Eye of the World (Wheel of Time, #1)'
recs = get_recommendations(df, indices, title, cosine_sim)
print(recs)