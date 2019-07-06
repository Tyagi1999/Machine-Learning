
# coding: utf-8

# In[31]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[32]:


data=pd.read_csv("Mall_Customers.csv")


# In[33]:


print(data)


# In[34]:


x=data.iloc[:,3:].values


# In[35]:


print(x)


# In[36]:


from sklearn.cluster import KMeans


# In[37]:


wcss=[]


# In[38]:


for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[39]:


print(wcss)


# In[40]:


plt.plot(range(1,11),wcss)
plt.title("Elbow method")
plt.xlabel("no. of cluters")
plt.ylabel("wcss")
plt.show()


# In[41]:


kmeans=KMeans(n_clusters=5,init="k-means++",random_state=0)


# In[42]:


y_kmeans=kmeans.fit_predict(x)


# In[43]:


print(y_means)


# In[44]:


plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=50,c="red",label="first")
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=50,c="blue",label="second")
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=50,c="green",label="third")
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=50,c="cyan",label="fourth")
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=50,c="magenta",label="fifth")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=60,c="yellow",label="ceteroids")
plt.title("k mens clustering")
plt.xlabel("income")
plt.ylabel("spending score")
plt.legend()
plt.show()

