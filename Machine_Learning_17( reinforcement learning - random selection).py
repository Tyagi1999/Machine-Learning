
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[20]:


data=pd.read_csv("Ads_CTR_Optimisation.csv")


# In[21]:


print(data)


# In[22]:


print(data.describe())


# In[23]:


import random


# In[24]:


N=10000
d=10
ads_selected=[]
total_reward=0


# In[25]:


for n in range(0,N):
    ad=random.randrange(d)
    ads_selected.append(ad)
    reward=data.values[n,ad]
    total_reward+=reward


# In[26]:


print(ads_selected)


# In[27]:


print(total_reward)


# In[28]:


plt.hist(ads_selected)
plt.title("ads selection randomly")
plt.xlabel("ads")
plt.ylabel("no. of times ad selected")
plt.show()


# In[29]:


counter=[]
for i in range(0,10):
    count=0
    for j in ads_selected:
        if i==j:
            count=count+1
    counter.append(count)        


# In[30]:


print(counter)

