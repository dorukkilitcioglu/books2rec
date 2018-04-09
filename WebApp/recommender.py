# Function that takes in book feature similarity matrices as input and outputs most similar book
def get_recommendations(bookid_to_title, title_to_bookid, title, similarities, weights):
    
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
    N = 20
    sim_scores = sim_scores[0:N]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar books
    recs = []
    for s in sim_scores:
        bookid = str(s[0] + 1)
        recs.append(bookid_to_title[bookid])
    return recs