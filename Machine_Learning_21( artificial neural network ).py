
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


data=pd.read_csv("Churn_Modelling.csv")


# In[3]:


print(data)


# In[5]:


data.describe()


# In[9]:


x=data.iloc[:,3:13].values
y=data.iloc[:,13].values


# In[10]:


print(x)


# In[11]:


print(y)


# In[12]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder


# In[13]:


lex1=LabelEncoder()


# In[14]:


x[:,1]=lex1.fit_transform(x[:,1])


# In[15]:


print(x[:,1])


# In[16]:


lex2=LabelEncoder()


# In[17]:


x[:,2]=lex2.fit_transform(x[:,2])


# In[18]:


print(x[:,2])


# In[20]:


ohe1=OneHotEncoder(categorical_features=[1])


# In[22]:


x=ohe1.fit_transform(x).toarray()


# In[23]:


x=x[:,1:]


# In[24]:


print(x)


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[27]:


from sklearn.preprocessing import StandardScaler


# In[28]:


sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[29]:


import keras


# In[30]:


from keras.models import Sequential


# In[32]:


from keras.layers import Dense


# In[33]:


classifier=Sequential()


# In[34]:


classifier.add(Dense(output_dim=6,input_dim=11,init="uniform",activation="relu"))


# In[35]:


classifier.add(Dense(output_dim=6,init="uniform",activation="relu"))


# In[36]:


classifier.add(Dense(output_dim=1,init="uniform",activation="sigmoid"))


# In[37]:


classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])


# In[38]:


classifier.fit(x_train,y_train,batch_size=10,nb_epoch=100)


# In[39]:


y_pred=classifier.predict(x_test)


# In[40]:


print(y_pred)


# In[41]:


print(y_test)


# In[42]:


y_pred=(y_pred>0.5)


# In[43]:


print(y_pred)


# In[45]:


from sklearn.metrics import confusion_matrix


# In[46]:


cm=confusion_matrix(y_pred,y_test)


# In[47]:


print(cm)


# In[49]:


print("accuracy of test is {0} %".format((2141/2500)*100))

