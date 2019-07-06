
# coding: utf-8

# In[20]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[21]:


data=pd.read_csv("Wine.csv")


# In[22]:


x=data.iloc[:,:13].values
y=data.iloc[:,13].values


# In[23]:


print(x)


# In[24]:


print(y)


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[27]:


from sklearn.preprocessing import StandardScaler


# In[28]:


sc=StandardScaler()


# In[29]:


x_train=sc.fit_transform(x_train)


# In[30]:


x_test=sc.transform(x_test)


# In[31]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# In[32]:


lda=LDA(n_components=2)


# In[33]:


x_train=lda.fit_transform(x_train,y_train)


# In[34]:


x_test=lda.transform(x_test)


# In[35]:


from sklearn.linear_model import LogisticRegression


# In[36]:


regressor=LogisticRegression()


# In[37]:


regressor.fit(x_train,y_train)


# In[38]:


y_pred=regressor.predict(x_test)


# In[39]:


print(y_pred)


# In[40]:


print(y_test)


# In[42]:


print(regressor.score(x_train,y_train))


# In[43]:


print(regressor.score(x_test,y_test))


# In[44]:


from sklearn.metrics import confusion_matrix


# In[45]:


cm=confusion_matrix(y_pred,y_test)


# In[46]:


print(cm)


# In[47]:


print("accuracy of test model is {0} %".format((45/45)*100))


# In[49]:


from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, regressor.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()


# In[50]:


from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, regressor.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()

