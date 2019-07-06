
# coding: utf-8

# In[39]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[40]:


data=pd.read_csv("Position_Salaries.csv")


# In[41]:


X=data.iloc[:,1:2].values
Y=data.iloc[:,2:].values


# In[42]:


print(X)


# In[43]:


print(Y)


# In[44]:


print(data.describe())


# In[45]:


print(data.head())


# In[46]:


print(data.tail())


# In[47]:


print(data['Position'])


# In[48]:


print(data['Position'][9])


# In[49]:


print(data['Salary'][9])


# In[50]:


from sklearn.preprocessing import StandardScaler


# In[51]:


sc_x=StandardScaler()
X=sc_x.fit_transform(X)
print(X)


# In[52]:


sc_y=StandardScaler()
Y=sc_y.fit_transform(Y)
print(Y)


# In[53]:


from sklearn.svm import SVR


# In[54]:


regressor=SVR(kernel='rbf')


# In[55]:


regressor.fit(X,Y)


# In[56]:


y_pred=sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))


# In[57]:


print(y_pred)


# In[59]:


plt.scatter(X,Y,color='r')
plt.plot(X,regressor.predict(X),color='b')
plt.title("SVR GRAPH")
plt.xlabel('level')
plt.ylabel('salary')
plt.show()

