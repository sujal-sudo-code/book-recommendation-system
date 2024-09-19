import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

book=pd.read_csv('Documents/Book Recomendetion System/Dataset/BX-Books.csv', sep=";", on_bad_lines='skip',encoding='latin-1')
book.head(2)
book.shape
book.columns

book=book[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher','Image-URL-L']]
book.head(2)
book.rename(columns={"Book-Title":"Title","Book-Author":"Author","Year-Of-Publication":"Year","Image-URL-L":"image-url"},inplace = True)
book.head(2)

users=pd.read_csv('Documents/Book Recomendetion System/Dataset/BX-Users.csv', sep=";", on_bad_lines='skip',encoding='latin-1')
users.head()
users.shape

ratings=pd.read_csv('Documents/Book Recomendetion System/Dataset/BX-Book-Ratings.csv', sep=";", on_bad_lines='skip',encoding='latin-1')
ratings.head()
ratings.shape

print(book.shape)
print(users.shape)
print(ratings.shape)

ratings['User-ID'].value_counts()
ratings['User-ID'].unique().shape
x=ratings["User-ID"].value_counts() > 200
x[x].shape
y=x[x].index

ratings=ratings[ratings['User-ID'].isin(y)]
ratings.head()
ratings.shape

book.head(2)

rating_with_books=ratings.merge(book,on="ISBN")
rating_with_books.head(3)
rating_with_books.shape
num_rating = rating_with_books.groupby('Title')['Book-Rating'].count().reset_index()
num_rating.head()
num_rating.rename(columns={"Book-Rating":"num_of_ratings"},inplace=True)
num_rating.head()
final_rating=rating_with_books.merge(num_rating,on='Title')
final_rating.head(2)
final_rating.shape
final_rating=final_rating[final_rating['num_of_ratings']>=50]
final_rating.head()
final_rating.sample(10)
final_rating.shape
final_rating.drop_duplicates(['User-ID','Title'],inplace=True)
final_rating.shape

book_pivert=final_rating.pivot_table(columns='User-ID',index='Title',values='Book-Rating')
book_pivert.
book_pivert.shape
book_pivert.fillna(0,inplace=True)
book_pivert

from scipy.sparse import csr_matrix
book_sparse=csr_matrix(book_pivert)
book_sparse

from sklearn.neighbors import NearestNeighbors
model=NearestNeighbors(algorithm='brute')
model.fit(book_sparse)
distance, suggestion =model.kneighbors(book_pivert.iloc[237,:].values.reshape(1,-1),n_neighbors=6)
distance
suggestion
for i in range(len(suggestion)):
    print(book_pivert.index[suggestion[i]])

book_pivert.index[3]
book_name = book_pivert.index
book_name

import pickle
pickle.dump(model, open('Documents/Book Recomendetion System/artifacts/model.pkl','wb'))
pickle.dump(book_name, open('Documents/Book Recomendetion System/artifacts/book_name.pkl','wb'))
pickle.dump(final_rating, open('Documents/Book Recomendetion System/artifacts/final_rating.pkl','wb'))
pickle.dump(book_pivert, open('Documents/Book Recomendetion System/artifacts/book_pivert .pkl','wb'))

def recommend_book(book_name):
    book_id=np.where(book_pivert.index==book_name)[0][0]
    distance, suggestion =model.kneighbors(book_pivert.iloc[book_id,:].values.reshape(1,-1),n_neighbors=6)
for i in range(len(suggestion)):
        book=book_pivert.index[suggestion[i]]
        for j in book:
            print(j)

book_name='Wish You Well'
recommend_book(book_name)
