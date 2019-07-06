
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


data=pd.read_csv("Breast_cancer_data.csv")


# In[3]:


X=data.iloc[:,0:5].values
Y=data.iloc[:,5].values


# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)


# In[7]:


from sklearn.preprocessing import StandardScaler


# In[8]:


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[9]:


from sklearn.naive_bayes import GaussianNB


# In[10]:


classifier=GaussianNB()


# In[11]:


classifier.fit(x_train,y_train)


# In[12]:


y_pred=classifier.predict(x_test)


# In[13]:


print(y_pred)


# In[14]:


print(y_test)


# In[15]:


from sklearn.metrics import confusion_matrix


# In[16]:


cm=confusion_matrix(y_test,y_pred)


# In[17]:


print(cm)


# In[18]:


print(classifier.score(x_train,y_train))


# In[19]:


print(classifier.score(x_test,y_test))


# In[21]:


print("Accuracy of model is {0} %".format((131/143)*100))


# In[22]:


plt.plot(x_train,classifier.predict(x_train),color='b')
plt.title("training graph")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# In[23]:


plt.plot(x_test,classifier.predict(x_test),color='b')
plt.title("test graph of predicted values")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# In[24]:


plt.plot(x_test,y_test,color='b')
plt.title("test  graph")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

