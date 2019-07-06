
# coding: utf-8

# In[112]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[113]:


dataset=pd.read_csv("insurance.csv")
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,6].values


# In[114]:


print("this is data X:")
print(X)
print("this is data Y:")
print(Y)


# In[115]:


from sklearn.cross_validation import train_test_split


# In[116]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.33,random_state=0)


# In[117]:


print("this is x_train:")
print(x_train)
print("this is x_test:")
print(x_test)
print("this is y_train:")
print(y_train)
print("this is y_test:")
print(y_test)


# In[118]:


from sklearn.preprocessing import StandardScaler


# In[119]:


sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
sc_y=StandardScaler()


# In[120]:


from sklearn.linear_model import LinearRegression


# In[121]:


regressor=LinearRegression()


# In[122]:


regressor.fit(x_train,y_train)


# In[123]:


y_pred=regressor.predict(x_test)


# In[124]:


print(y_pred)


# In[125]:


print(y_pred-y_test)


# In[126]:


dataset.describe()


# In[127]:


print(regressor.intercept_)
print(regressor.coef_)


# In[128]:


print((regressor.score(x_train,y_train))*100,"%")
print((regressor.score(x_test,y_test))*100,"%")


# In[129]:


import statsmodels.formula.api as sm


# In[130]:


X=np.append(arr=np.ones((348,1)).astype(int),values=X,axis=1)


# In[131]:


print(X)


# In[132]:


x_opt=X[:,[0,1,2,3,4,5,6]]
regressor_OLS=sm.OLS(endog=Y,exog=x_opt).fit()
regressor_OLS.summary()


# In[133]:


x_opt=X[:,[0,1,3,4,5,6]]
regressor_OLS=sm.OLS(endog=Y,exog=x_opt).fit()
regressor_OLS.summary()


# In[134]:


x_opt=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=x_opt).fit()
regressor_OLS.summary()


# In[144]:


x_opt=X[:,[0,1,3,5]]
regressor_OLS=sm.OLS(endog=Y,exog=x_opt).fit()
regressor_OLS.summary()


# In[145]:


print(x_opt)


# In[146]:


from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_opt, Y, test_size = 0.2, random_state = 0)


# In[147]:


print("this is x_train:")
print(x_train)
print("this is x_test:")
print(x_test)
print("this is y_train:")
print(y_train)
print("this is y_test:")
print(y_test)


# In[148]:


sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
sc_y=StandardScaler()


# In[149]:


regressor.fit(x_train,y_train)


# In[150]:


y_pred=regressor.predict(x_test)


# In[151]:


print(y_pred-y_test)


# In[152]:


print(regressor.intercept_)
print(regressor.coef_)


# In[156]:


print((regressor.score(x_train,y_train))*100,"%")
print((regressor.score(x_test,y_test))*100,"%")

