
# coding: utf-8

# In[42]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[43]:


data=pd.read_csv("Ads_CTR_Optimisation.csv")


# In[44]:


import math


# In[45]:


N=10000
d=10
ads_selected=[]
total_reward=0
no_of_selections=[0]*d
sum_of_reward=[0]*d


# In[46]:


print(no_of_selections)
print(sum_of_reward)


# In[56]:


for n in range(0,N):
    ad=0
    max_upper_bound=0
    for i in range(0,d):
        if no_of_selections[i] > 0:
            avg_reward=sum_of_reward[i]/no_of_selections[i]
            delta_i=math.sqrt(3/2*math.log(n+1)/no_of_selections[i])
            upper_bound=avg_reward+delta_i
        else:
            upper_bound=1e400
        if upper_bound > max_upper_bound:
            max_upper_bound=upper_bound
            ad=i
    ads_selected.append(ad)
    no_of_selections[ad]=no_of_selections[ad]+1
    reward=data.values[n,ad]
    sum_of_reward[ad]=sum_of_reward[ad]+reward
    total_reward=total_reward+reward


# In[57]:


print(ads_selected)


# In[58]:


counter=[]
for i in range(0,10):
    count=0
    for j in ads_selected:
        if i==j:
            count=count+1
    counter.append(count)        


# In[59]:


print(counter)


# In[60]:


print(total_reward)


# In[61]:


print(sum_of_reward)


# In[62]:


print(no_of_selections)


# In[63]:


print(ad)


# In[64]:


plt.hist(ads_selected)
plt.title("ads slections")
plt.xlabel("ads")
plt.ylabel("no of times ad selected")
plt.show()

