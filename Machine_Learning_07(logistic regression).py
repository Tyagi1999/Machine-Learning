
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


data=pd.read_csv('Breast_cancer_data.csv')


# In[17]:


X=data.iloc[:,0:5].values
Y=data.iloc[:,5].values


# In[18]:


print(X)


# In[19]:


print(Y)


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)


# In[22]:


print("x train: ")
print(x_train)
print("x test: ")
print(x_test)
print("y train: ")
print(y_train)
print("y test: ")
print(y_test)


# In[23]:


from sklearn.preprocessing import StandardScaler


# In[24]:


sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)


# In[25]:


print("training X")
print(x_train)
print("testing X")
print(x_test)


# In[26]:


from sklearn.linear_model import LogisticRegression


# In[27]:


regressor=LogisticRegression()


# In[28]:


regressor.fit(x_train,y_train)


# In[29]:


y_pred=regressor.predict(x_test)


# In[30]:


print(y_pred)


# In[34]:


from sklearn.metrics import confusion_matrix


# In[35]:


cm=confusion_matrix(y_test,y_pred)


# In[36]:


print(cm)


# In[37]:


print(regressor.score(x_train,y_train))


# In[38]:


print(regressor.score(x_test,y_test))


# In[45]:


print("Accuracy of model is: {0} %".format(132/143*100))


# In[48]:



plt.plot(x_train,regressor.predict(x_train),color='b')
plt.title("training graph")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# In[62]:


plt.plot(x_test,regressor.predict(x_test),color='b')
plt.title("test graph of predicted values")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# In[63]:


plt.plot(x_test,y_test,color='b')
plt.title("test  graph")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

