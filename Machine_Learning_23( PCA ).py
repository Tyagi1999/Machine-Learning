
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[4]:


data=pd.read_csv("Wine.csv")


# In[5]:


print(data)


# In[6]:


print(data.describe())


# In[7]:


x=data.iloc[:,:13].values
y=data.iloc[:,13].values


# In[8]:


print(x)


# In[9]:


print(y)


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[12]:


print(x_train)


# In[13]:


print(x_test)


# In[14]:


print(y_train)


# In[15]:


print(y_test)


# In[16]:


from sklearn.preprocessing import StandardScaler


# In[17]:


sc=StandardScaler()


# In[18]:


x_train=sc.fit_transform(x_train)


# In[19]:


x_test=sc.transform(x_test)


# In[20]:


from sklearn.decomposition import PCA


# In[21]:


pca=PCA(n_components=2)


# In[22]:


x_train=pca.fit_transform(x_train)


# In[23]:


x_test=pca.transform(x_test)


# In[24]:


explained_variance=pca.explained_variance_ratio_


# In[25]:


print(explained_variance)


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


# In[31]:


print(y_test)


# In[32]:


from sklearn.metrics import confusion_matrix


# In[33]:


cm=confusion_matrix(y_pred,y_test)


# In[34]:


print(cm)


# In[35]:


print(regressor.score(x_train,y_train))


# In[36]:


print(regressor.score(x_test,y_test))


# In[37]:


print("accuracy of test is {0} %".format(44/45))


# In[38]:


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
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


# In[39]:


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
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

