
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


data=pd.read_csv("Wine.csv")
x=data.iloc[:,:13].values
y=data.iloc[:,13].values


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[7]:


from sklearn.preprocessing import StandardScaler


# In[8]:


sc=StandardScaler()


# In[9]:


x_train=sc.fit_transform(x_train)


# In[10]:


x_test=sc.transform(x_test)


# In[11]:


from sklearn.decomposition import KernelPCA


# In[12]:


kpca=KernelPCA(kernel="rbf",n_components=2)


# In[13]:


x_train=kpca.fit_transform(x_train)


# In[14]:


x_test=kpca.transform(x_test)


# In[15]:


from sklearn.linear_model import LogisticRegression


# In[16]:


regressor=LogisticRegression()


# In[17]:


regressor.fit(x_train,y_train)


# In[19]:


y_pred=regressor.predict(x_test)


# In[20]:


print(y_pred)


# In[21]:


print(y_test)


# In[22]:


print(regressor.score(x_train,y_train))


# In[23]:


print(regressor.score(x_test,y_test))


# In[24]:


from sklearn.metrics import confusion_matrix


# In[25]:


cm=confusion_matrix(y_pred,y_test)


# In[26]:


print(cm)


# In[27]:


print("accuracy of test model is {0} %".format((45/45)*100))


# In[30]:


from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, regressor.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red','green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[31]:


from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, regressor.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red','green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

