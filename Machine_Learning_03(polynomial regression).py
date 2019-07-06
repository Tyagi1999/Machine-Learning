
# coding: utf-8

# In[1]:


import pandas  as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


data=pd.read_csv("Position_Salaries.csv")


# In[3]:


data.describe()


# In[6]:


X=data.iloc[:,1:2].values
Y=data.iloc[:,2].values


# In[7]:


print("this is X: ")
print(X)
print("this is Y: ")
print(Y)


# In[8]:


from sklearn.linear_model import LinearRegression


# In[9]:


regressor1=LinearRegression()


# In[12]:


regressor1.fit(X,Y)


# In[13]:


from sklearn.preprocessing import PolynomialFeatures


# In[55]:


poly=PolynomialFeatures(degree=4)


# In[56]:


X_poly=poly.fit_transform(X)


# In[57]:


print(X_poly)


# In[58]:


poly.fit(X_poly,Y)


# In[59]:


regressor2=LinearRegression()


# In[60]:


regressor2.fit(X_poly,Y)


# In[69]:


print("accuracy before polynomial regression: ")
print(regressor1.score(X,Y)*100,"%")


# In[70]:


print("accuracy after polynomial regression: ")
print(regressor2.score(X_poly,Y)*100,"%")


# In[71]:


print(regressor1.predict(6.5))


# In[72]:


print(regressor2.predict(poly.fit_transform(6.5)))


# In[73]:


plt.scatter(X,Y,color='r')
plt.plot(X,regressor1.predict(X),color='b')
plt.title('salary checking')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()


# In[74]:


plt.scatter(X,Y,color='g')
plt.plot(X,regressor2.predict(poly.fit_transform(X)),color='b')
plt.title('salary checking')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

