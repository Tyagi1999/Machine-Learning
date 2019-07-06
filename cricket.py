
# coding: utf-8

# In[54]:


import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np


# In[55]:


data=pd.read_csv("Book1.csv")


# In[56]:


print(data)


# In[57]:


century=pd.read_csv("Book2(Century).csv")


# In[58]:


print(century)


# In[59]:


half_cent=pd.read_csv("Book3(HC).csv")


# In[60]:


print(half_cent)


# In[61]:


x=data.iloc[:,:1].values


# In[62]:


print(x)


# In[63]:


y=data.iloc[:,1:].values


# In[64]:


print(y)


# In[65]:


import scipy as sp


# In[66]:


sp.sum(sp.isnan(y))


# In[67]:


y[np.isnan(y)]=-1


# In[68]:


print(y)


# In[69]:


y1=century.iloc[:,1:].values


# In[70]:


y1[np.isnan(y1)]=-1


# In[71]:


print(y1)


# In[72]:


y2=half_cent.iloc[:,1:].values


# In[73]:


y2[np.isnan(y2)]=-1


# In[74]:


print(y2)


# In[75]:


a=input("enter player name: ")
b=int(input("enter the year: "))


# In[76]:


dic={2006:0,2007:1,2008:2,2009:3,2010:4,2011:5,2012:6,2013:7,2014:8,2015:9,2016:10,2017:11,2018:12}  


# In[77]:


print(np.size(y))


# In[78]:


for i in range(0,np.size(x)):
    if a==x[i]:
        index=i
score=y[index][dic[b]]
if score==-1:
    run="Not Played"
else:
    run=score
no_of_c=y1[index][dic[b]]
no_of_hc=y2[index][dic[b]]
total_runs=0
for i in range(0,dic[b+1]):
    if y[index][i]==-1:
        total_runs+=0
    else:
        total_runs+=y[index][dic[b]]        


# In[79]:


print("Total runs scored by {0} are {1}".format(a,run))
print("Cummulative runs scored upto year {0} are {1}".format(b,total_runs))
print("Total no of centuries and half centuries in {0} are {1} and {2} respectively".format(b,no_of_c,no_of_hc))

