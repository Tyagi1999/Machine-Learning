
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


data=pd.read_csv("Position_Salaries.csv")


# In[4]:


data.describe()


# In[5]:


print(data)


# In[12]:


X=data.iloc[:,1:2].values
Y=data.iloc[:,2:].values


# In[13]:


print(X)


# In[14]:


print(Y)


# In[16]:


from sklearn.ensemble import RandomForestRegressor


# In[18]:


regressor=RandomForestRegressor(n_estimators=300,random_state=0)


# In[19]:


regressor.fit(X,Y)


# In[20]:


y_pred=regressor.predict(6.5)


# In[21]:


print(y_pred)


# In[22]:


plt.scatter(X,Y,color='r')
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.plot(X_grid, regressor.predict(X_grid), color = 'b')
plt.title('RANDOM FOREST REGRESSOR')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

