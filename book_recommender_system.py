    #!/usr/bin/env python
    # coding: utf-8

    # In[5]:


    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns


    # In[154]:


    book=pd.read_csv('Documents/Book Recomendetion System/Dataset/BX-Books.csv', sep=";", on_bad_lines='skip',encoding='latin-1')


    # In[69]:


    book.head(2)


    # In[70]:


    book.shape


    # In[71]:


    book.columns


    # In[72]:


    book=book[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher',
        'Image-URL-L']]


    # In[73]:


    book.head(2)


    # In[74]:


    book.rename(columns={
        "Book-Title":"Title",
        "Book-Author":"Author",
        "Year-Of-Publication":"Year",
        "Image-URL-L":"image-url"},inplace = True)


    # In[75]:


    book.head(2)


    # In[155]:


    users=pd.read_csv('Documents/Book Recomendetion System/Dataset/BX-Users.csv', sep=";", on_bad_lines='skip',encoding='latin-1')


    # In[156]:


    users.head()


    # In[78]:


    users.shape


    # In[158]:


    ratings=pd.read_csv('Documents/Book Recomendetion System/Dataset/BX-Book-Ratings.csv', sep=";", on_bad_lines='skip',encoding='latin-1')


    # In[159]:


    ratings.head()


    # In[81]:


    ratings.shape


    # In[82]:


    print(book.shape)
    print(users.shape)
    print(ratings.shape)


    # In[83]:


    ratings['User-ID'].value_counts()


    # In[84]:


    ratings['User-ID'].unique().shape


    # In[85]:


    x=ratings["User-ID"].value_counts() > 200


    # In[86]:


    x[x].shape


    # In[87]:


    y=x[x].index


    # In[88]:


    y


    # In[89]:


    ratings=ratings[ratings['User-ID'].isin(y)]


    # In[90]:


    ratings.head()


    # In[91]:


    ratings.shape


    # In[92]:


    book.head(2)


    # In[94]:


    rating_with_books=ratings.merge(book,on="ISBN")


    # In[96]:


    rating_with_books.head(3)


    # In[97]:


    rating_with_books.shape


    # In[101]:


    num_rating = rating_with_books.groupby('Title')['Book-Rating'].count().reset_index()


    # In[102]:


    num_rating.head()


    # In[105]:


    num_rating.rename(columns={"Book-Rating":"num_of_ratings"},inplace=True)


    # In[106]:


    num_rating.head()


    # In[109]:


    final_rating=rating_with_books.merge(num_rating,on='Title')


    # In[111]:


    final_rating.head(2)


    # In[112]:


    final_rating.shape


    # In[115]:


    final_rating=final_rating[final_rating['num_of_ratings']>=50]


    # In[118]:


    final_rating.head()


    # In[120]:


    final_rating.sample(10)


    # In[121]:


    final_rating.shape


    # In[123]:


    final_rating.drop_duplicates(['User-ID','Title'],inplace=True)


    # In[124]:


    final_rating.shape


    # In[126]:


    book_pivert=final_rating.pivot_table(columns='User-ID',index='Title',values='Book-Rating')


    # In[127]:


    book_pivert.


    # In[129]:


    book_pivert.shape


    # In[130]:


    book_pivert.fillna(0,inplace=True)


    # In[131]:


    book_pivert


    # In[132]:


    from scipy.sparse import csr_matrix


    # In[133]:


    book_sparse=csr_matrix(book_pivert)


    # In[134]:


    book_sparse


    # In[135]:


    from sklearn.neighbors import NearestNeighbors
    model=NearestNeighbors(algorithm='brute')


    # In[136]:


    model.fit(book_sparse)


    # In[147]:


    distance, suggestion =model.kneighbors(book_pivert.iloc[237,:].values.reshape(1,-1),n_neighbors=6)


    # In[141]:


    distance


    # In[142]:


    suggestion


    # In[148]:


    for i in range(len(suggestion)):
        print(book_pivert.index[suggestion[i]])


    # In[150]:


    book_pivert.index[3]


    # In[151]:


    book_name = book_pivert.index


    # In[152]:


    book_name


    # In[162]:


    import pickle
    pickle.dump(model, open('Documents/Book Recomendetion System/artifacts/model.pkl','wb'))
    pickle.dump(book_name, open('Documents/Book Recomendetion System/artifacts/book_name.pkl','wb'))
    pickle.dump(final_rating, open('Documents/Book Recomendetion System/artifacts/final_rating.pkl','wb'))
    pickle.dump(book_pivert, open('Documents/Book Recomendetion System/artifacts/book_pivert .pkl','wb'))


    # In[165]:


    def recommend_book(book_name):
        book_id=np.where(book_pivert.index==book_name)[0][0]
        distance, suggestion =model.kneighbors(book_pivert.iloc[book_id,:].values.reshape(1,-1),n_neighbors=6)
        
        for i in range(len(suggestion)):
            book=book_pivert.index[suggestion[i]]
            for j in book:
                print(j)


    # In[168]:


    book_name='Wish You Well'
    recommend_book(book_name)


    # In[ ]:




