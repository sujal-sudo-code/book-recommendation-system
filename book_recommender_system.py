import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

book=pd.read_csv('C:\Users\acer\Documents\Book Recomendetion System\Dataset\BX-Books.csv', sep=";", on_bad_lines='skip',encoding='latin-1')

book.head(2)

book.shape

book.columns

book=book[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher',
       'Image-URL-L']]

book.head(2)

book.rename(columns={
    "Book-Title":"Title",
    "Book-Author":"Author",
    "Year-Of-Publication":"Year",
    "Image-URL-L":"image-url"},inplace = True)

book.head(2)

users=pd.read_csv('C:\Users\acer\Documents\Book Recomendetion System\Dataset\BX-Users.csv', sep=";", on_bad_lines='skip',encoding='latin-1')

users.head()

 users.shape

ratings=pd.read_csv('C:\Users\acer\Documents\Book Recomendetion System\Dataset\BX-Book-Ratings.csv', sep=";", on_bad_lines='skip',encoding='latin-1')

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

y

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

book_pivert

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

book_name='Harry Potter and the Prisoner of Azkaban (Book 3)'
recommend_book(book_name)

from sklearn.model_selection import train_test_split

train, test = train_test_split(final_rating, test_size=0.2, random_state=42)

train_pivert = train.pivot_table(columns='User-ID', index='Title', values='Book-Rating').fillna(0)
test_pivert = test.pivot_table(columns='User-ID', index='Title', values='Book-Rating').fillna(0)

ratings.head()

train_sparse = csr_matrix(train_pivert)

model.fit(train_sparse)

def precision_recall_at_k(k=5):
    precision_list = []
    recall_list = []
    
    for user_id in test['User-ID'].unique():
        user_ratings = test[test['User-ID'] == user_id]
        user_books_rated_high = user_ratings[user_ratings['Book-Rating'] >= 4]['Title'].values
        
        if len(user_books_rated_high) == 0:
            continue
            
        if user_id in train_pivert.columns:
            book_ratings = train_pivert[user_id]
            recommended_books = book_ratings.sort_values(ascending=False).head(k).index.values
            relevant_and_recommended = [book for book in recommended_books if book in user_books_rated_high]
            precision = len(relevant_and_recommended) / len(recommended_books) if len(recommended_books) > 0 else 0
            recall = len(relevant_and_recommended) / len(user_books_rated_high) if len(user_books_rated_high) > 0 else 0
            precision_list.append(precision)
            recall_list.append(recall)
    
    avg_precision = np.mean(precision_list) if precision_list else 0
    avg_recall = np.mean(recall_list) if recall_list else 0
    return avg_precision, avg_recall

precision_at_450, recall_at_450 = precision_recall_at_k(k=450)
if (precision_at_450 + recall_at_450) > 0:
    f1_score_at_450 = 2 * (precision_at_450 * recall_at_450) / (precision_at_450 + recall_at_450)
else:
    f1_score_at_450 = 0

print(f"Precision @ 450: {precision_at_450:.4f}")
print(f"Recall @ 450: {recall_at_450:.4f}")
print(f"F1 Score @ 450: {f1_score_at_450:.4f}")


def hit_rate_accuracy_at_k(k=450):
    total_users = 0
    successful_hits = 0
    
    for user_id in test['User-ID'].unique():
        user_ratings = test[test['User-ID'] == user_id]
        user_books_rated_high = user_ratings[user_ratings['Book-Rating'] >= 4]['Title'].values  
        
        if len(user_books_rated_high) == 0:
            continue
        
        total_users += 1
        
        if user_id in train_pivert.columns:
            book_ratings = train_pivert[user_id]
            recommended_books = book_ratings.sort_values(ascending=False).head(k).index.values  
            
            if any(book in user_books_rated_high for book in recommended_books):
                successful_hits += 1

    accuracy_percentage = (successful_hits / total_users) * 100 if total_users > 0 else 0
    return accuracy_percentage

accuracy_at_450 = hit_rate_accuracy_at_k(k=450)
print(f"Accuracy @ 450: {accuracy_at_450:.2f}%")

def calculate_confusion_matrix_at_k(k=450):
    tp, fp, tn, fn = 0, 0, 0, 0
    
    for user_id in test['User-ID'].unique():
        user_ratings = test[test['User-ID'] == user_id]
        user_books_rated_high = user_ratings[user_ratings['Book-Rating'] >= 4]['Title'].values
        
        if len(user_books_rated_high) == 0:
            continue
            
        if user_id in train_pivert.columns:
            book_ratings = train_pivert[user_id]
            recommended_books = book_ratings.sort_values(ascending=False).head(k).index.values
            
            for book in recommended_books:
                if book in user_books_rated_high:
                    tp += 1  
                else:
                    fp += 1  
            
            for book in user_books_rated_high:
                if book not in recommended_books:
                    fn += 1  
    tn = len(test) - (tp + fp + fn)
    
    return tp, fp, tn, fn


def plot_confusion_matrix(tp, fp, tn, fn, k):  
    confusion_matrix = np.array([[tp, fn],
                                 [fp, tn]])
    
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Recommended', 'Not Recommended'],
                yticklabels=['Relevant', 'Not Relevant'])
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix @ Top-{k} Recommendations')  
    plt.show()

tp, fp, tn, fn = calculate_confusion_matrix_at_k(k=450)
plot_confusion_matrix(tp, fp, tn, fn, k=450)


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = load_iris()
X = data.data
y = data.target

model = DecisionTreeClassifier()
model.fit(X, y)

plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=data.feature_names, 
           class_names=data.target_names.tolist(), rounded=True)  
plt.title("Decision Tree Visualization")
plt.show()


