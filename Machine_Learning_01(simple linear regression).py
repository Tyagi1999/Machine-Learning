
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[6]:


data=pd.read_csv("Salary.csv")
X=data.iloc[:,:-1].values
print(X)
Y=data.iloc[:,1].values
print(Y)


# In[7]:


from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=0)


# In[49]:


print(x_train)
print(x_test)
print(y_train)
print(y_test)


# In[50]:


data.describe()


# In[51]:


from sklearn.linear_model import LinearRegression


# In[52]:


regression=LinearRegression()


# In[53]:


regression.fit(x_train,y_train)


# In[54]:


y_pred=regression.predict(x_test)


# In[55]:


print(y_pred)


# In[56]:


print(y_test-y_pred)


# In[57]:


plt.scatter(x_train,y_train,color='r')
plt.plot(x_train,regression.predict(x_train),color='b')
plt.xlabel("experince")
plt.ylabel("salary")
plt.title("train graph(exp. vs salary)")
plt.show()


# In[58]:


plt.scatter(x_test,y_test,color='g')
plt.plot(x_test,regression.predict(x_test),color='magenta')
plt.xlabel("experience")
plt.ylabel("salary")
plt.title("test graph(exp. vs salary)")
plt.show()


# In[59]:


print(regression.intercept_)
print(regression.coef_)


# In[66]:


print((regression.score(x_test,y_test))*100,"%")


# In[67]:


print((regression.score(x_train,y_train))*100,"%")

