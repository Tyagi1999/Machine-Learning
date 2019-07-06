
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[12]:


data=pd.read_csv("Restaurant_Reviews.tsv",delimiter="\t")


# In[13]:


print(data)


# In[14]:


import re
import nltk
nltk.download("stopwords")


# In[16]:


from nltk.corpus import stopwords


# In[17]:


from nltk.stem.porter import PorterStemmer


# In[18]:


corpus=[]


# In[20]:


for i in range(0,1000):
    review=re.sub("[^a-zA-Z]"," ",data["Review"][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
    review=" ".join(review)
    corpus.append(review)


# In[22]:


print(review)


# In[23]:


print(corpus)


# In[24]:


from sklearn.feature_extraction.text import CountVectorizer


# In[25]:


cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()
y=data.iloc[:,1].values


# In[26]:


print(x)


# In[27]:


print(y)


# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[30]:


from sklearn.naive_bayes import GaussianNB


# In[31]:


classifier=GaussianNB()
classifier.fit(x_train,y_train)


# In[32]:


y_pred=classifier.predict(x_test)


# In[33]:


print(y_pred)


# In[34]:


print(y_test)


# In[36]:


print(classifier.score(x_train,y_train))


# In[37]:


print(classifier.score(x_test,y_test))


# In[38]:


from sklearn.metrics import confusion_matrix


# In[39]:


cm=confusion_matrix(y_pred,y_test)


# In[40]:


print(cm)


# In[43]:


print("accuracy is {0} %".format((67+113)*100/250))

