
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[3]:


data=pd.read_csv('Position_Salaries.csv')


# In[4]:


X=data.iloc[:,1:2].values
Y=data.iloc[:,2:].values


# In[5]:


print(X)


# In[6]:


print(Y)


# In[8]:


from sklearn.tree import DecisionTreeRegressor


# In[9]:


regressor=DecisionTreeRegressor(random_state=0)


# In[10]:


regressor.fit(X,Y)


# In[11]:


y_pred=regressor.predict(6.5)


# In[12]:


print(y_pred)


# In[18]:


plt.scatter(X,Y,color='r')
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.plot(X_grid, regressor.predict(X_grid), color = 'b')
plt.title('DECISION TREE REGRESSOR')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()


# In[20]:


print(X_grid)

