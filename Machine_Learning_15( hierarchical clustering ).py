
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


data=pd.read_csv("Mall_Customers.csv")


# In[3]:


x=data.iloc[:,3:].values


# In[4]:


import scipy.cluster.hierarchy as sch


# In[6]:


dendo=sch.dendrogram(sch.linkage(x,method="ward"))
plt.title("Dendogram")
plt.xlabel("customers")
plt.ylabel("eucledian distance")
plt.show()


# In[7]:


from sklearn.cluster import AgglomerativeClustering


# In[12]:


hc=AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")


# In[13]:


y_hc=hc.fit_predict(x)


# In[14]:


print(y_hc)


# In[16]:


plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=50,c="red",label="first")
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=50,c="blue",label="second")
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=50,c="green",label="third")
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=50,c="cyan",label="fourth")
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=50,c="magenta",label="fifth")
plt.title("Hierarchical clustering")
plt.xlabel("income")
plt.ylabel("spending score")
plt.legend()
plt.show()

