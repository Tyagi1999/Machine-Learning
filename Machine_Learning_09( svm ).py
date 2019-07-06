
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv("Breast_cancer_data.csv")


# In[3]:


X=data.iloc[:,0:5].values
Y=data.iloc[:,5].values


# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)


# In[40]:


from sklearn.preprocessing import StandardScaler


# In[41]:


sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)


# In[42]:


# support vector machine(SVM)
from sklearn.svm import SVC


# In[43]:


classifier=SVC(kernel="linear",random_state=0)


# In[44]:


classifier.fit(x_train,y_train)


# In[45]:


y_pred=classifier.predict(x_test)


# In[46]:


print(y_pred)


# In[47]:


print(y_test)


# In[48]:


from sklearn.metrics import confusion_matrix


# In[49]:


cm=confusion_matrix(y_test,y_pred)


# In[50]:


print(cm)


# In[51]:


print(classifier.score(x_train,y_train))


# In[52]:


print(classifier.score(x_test,y_test))


# In[53]:


print("Accuracy of model is: {0} %".format(132/143*100))


# In[54]:


plt.plot(x_train,classifier.predict(x_train),color='b')
plt.title("training graph")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# In[55]:


plt.plot(x_test,classifier.predict(x_test),color='b')
plt.title("test graph of predicted values")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# In[56]:


plt.plot(x_test,y_test,color='b')
plt.title("test  graph")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

