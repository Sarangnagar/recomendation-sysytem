#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
import warnings


# In[23]:


warnings.filterwarnings('ignore')


# In[52]:


columns_names =["user_id","item_id","rating","timestamp"]
df1 = pd.read_csv("u.data",sep='\t',names=columns_names)


# In[53]:


df1.head()


# In[54]:


df1.shape


# In[55]:


df1['user_id'].nunique()


# In[56]:


df1['item_id'].nunique()


# In[ ]:





# In[ ]:





# In[ ]:





# In[57]:


try:
    # read the CSV file into a pandas dataframe
    df = pd.read_csv("u.item",sep='\|', encoding='utf-8',header=None)
    
except UnicodeDecodeError:
    # if an exception occurs, try reading the file using the ISO-8859-1 encoding
    df = pd.read_csv("u.item",sep='\|', encoding='iso-8859-1',header=None)

# print the first 5 rows of the dataframe
print(df.head())


# In[58]:


df.shape


# In[59]:


df=df[[0,1]]
df.columns=['item_id','title']


# In[60]:


df.head()


# In[63]:


df2 = pd.merge(df1,df,on="item_id")


# In[64]:


df2.tail()


# In[65]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')


# In[70]:


df2.groupby('title').mean()['rating'].sort_values(ascending=False).head()


# In[73]:


df2.groupby('title').count()['rating'].sort_values(ascending=False)


# In[76]:


ratings = pd.DataFrame(df2.groupby('title').mean()['rating'])


# In[77]:


ratings.head()


# In[78]:


ratings['num of ratings'] = pd.DataFrame(df2.groupby('title').count()['rating'])


# In[79]:


ratings


# In[81]:


ratings.sort_values(by ='rating',ascending =False)


# In[83]:


plt.figure(figsize=(10,6))
plt.hist(ratings['num of ratings'],bins=70)
plt.show()


# In[84]:


plt.figure(figsize=(10,6))
plt.hist(ratings['rating'],bins=70)
plt.show()


# In[88]:


sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)


# In[90]:


df2.head()


# In[92]:


moviemat =df2.pivot_table(index="user_id",columns="title",values="rating")


# In[93]:


moviemat


# In[94]:


ratings.sort_values('num of ratings',ascending = False).head()


# In[97]:


starwars_user_ratings =moviemat['Star Wars (1977)']
starwars_user_ratings.head()


# In[99]:


similar_star_wars =moviemat.corrwith(starwars_user_ratings)
similar_star_wars


# In[117]:


corr_starwars =pd.DataFrame(similar_star_wars,columns =['Correlation'])
corr_starwars


# In[118]:


corr_starwars.dropna(inplace = True)
corr_starwars


# In[119]:


corr_starwars.head()


# In[120]:


corr_starwars.sort_values('Correlation',ascending=False).head(10)


# In[121]:


ratings


# In[122]:


corr_starwars=corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()


# In[124]:


corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False)


# In[128]:


def predict_movies(movie_name):
    movie_user_ratings=moviemat[movie_name]
    similar_to_movie=moviemat.corrwith(movie_user_ratings)
    corr_movie =pd.DataFrame(similar_to_movie,columns =['Correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie=corr_movie.join(ratings['num of ratings'])
    predictions =corr_movie[corr_movie['num of ratings']>100].sort_values('Correlation',ascending=False)
    return predictions
    


# In[129]:


predictions = predict_movies("Titanic (1997)")


# In[130]:


predictions.head()


# In[ ]:




