
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


data=pd.read_csv("Ads_CTR_Optimisation.csv")


# In[5]:


import math
import random


# In[6]:


N=10000
d=10
ads_selected=[]
no_of_reward_1=[0]*d
no_of_reward_0=[0]*d
total_reward=0


# In[7]:


for n in range(0,N):
    ad=0
    max_random=0
    for i in range(0,d):
        random_beta=random.betavariate(no_of_reward_1[i]+1,no_of_reward_0[i]+1)
        print(random_beta)
        if random_beta>max_random:
            max_random=random_beta
            ad=i
    ads_selected.append(ad)
    reward=data.values[n,ad]
    if reward==1:
        no_of_reward_1[ad]=no_of_reward_1[ad]+1
    else:
        no_of_reward_0[ad]=no_of_reward_0[ad]+1
    total_reward=total_reward+reward    


# In[8]:


print(ads_selected)


# In[9]:


print(ad)


# In[10]:


print(total_reward)


# In[11]:


plt.hist(ads_selected)
plt.title("ads selection")
plt.xlabel("ads")
plt.ylabel("no of times ad selected")
plt.show()

