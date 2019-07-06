
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


# In[6]:


from sklearn.preprocessing import StandardScaler


# In[7]:


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[8]:


from sklearn.ensemble import RandomForestClassifier


# In[10]:


classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)


# In[11]:


y_pred=classifier.predict(x_test)


# In[12]:


print(y_pred)


# In[13]:


print(y_test)


# In[14]:


from sklearn.metrics import confusion_matrix


# In[15]:


cm=confusion_matrix(y_pred,y_test)


# In[16]:


print(cm)


# In[17]:


print(classifier.score(x_train,y_train))


# In[18]:


print(classifier.score(x_test,y_test))


# In[19]:


print("Accuracy of model is {0} %".format((133/143)*100))


# In[20]:


plt.plot(x_train,classifier.predict(x_train),color='b')
plt.title("training graph")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# In[21]:


plt.plot(x_test,classifier.predict(x_test),color='b')
plt.title("test graph of predicted values")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# In[22]:


plt.plot(x_test,y_test,color='b')
plt.title("test  graph")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

