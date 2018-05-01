import os
import sys
import numpy as np
import pandas as pd
import scipy

# Function that takes in book feature similarity matrices as input and outputs most similar book
def get_recommendations(books, bookid_to_title, title_to_bookid, title, similarities, weights):
    
    # Get the index of the book that matches the title
    idx = int(title_to_bookid[title])
    idx -= 1
    
    # Get the total number of books
    num_books = len(similarities[0])

    # Get the pairwsie similarity scores of all books with that book
    similarity_scores = []
    for similarity in similarities:
        similarity_scores.append(list(enumerate(similarity[idx])))
    
    # Sum and average the similarity scores of the three feature sets to get true similarity
    sim_scores = []
    for i in range(num_books):  
        book_id = similarity_scores[0][i][0]
        
        score = 0
        for j in range(len(weights)):
            score += (similarity_scores[j][i][1] * weights[j])
            
        sim_scores.append((book_id, score))
        
    # Sort the books based on the highest similarity scores first
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the N most similar books
    N = 31
    sim_scores = sim_scores[1:N]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar books
    top_books = []
    for i in range(len(sim_scores)):
        s = sim_scores[i]
        book_id = s[0]
        book = books.iloc[book_id]
        book['rank'] = i + 1

        # for some reason, some of the text fields have newlines appended to them
        book['title'] = book['title'].strip()
        book['author'] = book['author'].strip()
        top_books.append(book)

    # chunk into groups of 3 to display better in web app
    chunks = []
    current_chunk = []
    for i in range(len(top_books)):
        if len(current_chunk) < 3:
            current_chunk.append(top_books[i])
        else:
            chunks.append(current_chunk)
            current_chunk = [top_books[i]]

    chunks.append(current_chunk)
    return chunks

def get_top_n_recs(result, books, n, q):
    recs = []
    for i in range(len(result)):
        if q[i] == 0: # book user hasn't already rated
            recs.append((i, result[i]))
        else:
            recs.append((i, float('-inf')))
    recs = sorted(recs, key=lambda tup: tup[1], reverse=True)

    top_books = []
    for i in range(n):
        book_id = recs[i][0]
        book = books.iloc[book_id]
        book['rank'] = i + 1

        # for some reason, some of the text fields have newlines appended to them
        book['title'] = book['title'].strip()
        book['author'] = book['author'].strip()
        top_books.append(book)
    
    # chunk into groups of 3 to display better in web app
    chunks = []
    current_chunk = []
    for i in range(len(top_books)):
        if len(current_chunk) < 3:
            current_chunk.append(top_books[i])
        else:
            chunks.append(current_chunk)
            current_chunk = [top_books[i]]
    chunks.append(current_chunk)

    return chunks

def map_user(q, V):
    # map new user to concept space by q*V
    user_to_concept = np.matmul(q, V)
    # map user back to itme space with user_to_concept * VT
    result = np.matmul(user_to_concept, V.T)
    return result

def map_user_sparse(q, V):
    q_sparse = scipy.sparse.csr_matrix(q)
    # map new user to concept space by q*V
    user_to_concept = q_sparse.dot(V)
    # map user back to itme space with user_to_concept * VT
    result = user_to_concept.dot(V.T).todense()
    return result.T

def most_popular(books, n):
    top_books = []
    for i in range(n):
        book = books.iloc[i]
        book['rank'] = i + 1

        # for some reason, some of the text fields have newlines appended to them
        book['title'] = book['title'].strip()
        book['author'] = book['author'].strip()
        top_books.append(book)

    # chunk into groups of 4 to display better in web app
    chunks = []
    current_chunk = []
    for i in range(len(top_books)):
        if len(current_chunk) < 3:
            current_chunk.append(top_books[i])
        else:
            chunks.append(current_chunk)
            current_chunk = [top_books[i]]
    chunks.append(current_chunk)

    return chunks