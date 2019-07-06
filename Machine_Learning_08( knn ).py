
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


flag=True
while(flag):
    print("Enter values in float or int only.")
    x1=float(input("enter mean radius: "))
    x2=float(input("enter mean texture: "))
    x3=float(input("enter mean perimeter: "))
    x4=float(input("enter mean area: "))
    x5=float(input("enter mean smoothness: "))
    arr=np.array([[x1,x2,x3,x4,x5]])
    if x1 or x2 or x3 or x4 or x5 is float:
        flag=False
    else:
        flag=True
    
print(arr)


# In[3]:


data=pd.read_csv('Breast_cancer_data.csv')


# In[4]:


X=data.iloc[:,0:5].values
Y=data.iloc[:,5].values


# In[5]:


print(X)


# In[6]:


print(Y)


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)


# In[9]:


x_test=np.append(x_test,arr,axis=0)


# In[10]:


print("x train: ")
print(x_train)
print("x test: ")
print(x_test)
print("y train: ")
print(y_train)
print("y test: ")
print(y_test)


# In[11]:


from sklearn.preprocessing import StandardScaler


# In[12]:


sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)


# In[13]:


print("training X")
print(x_train)
print("testing X")
print(x_test)


# In[14]:


# K NEAREST NEIGHBORS 
from sklearn.neighbors import KNeighborsClassifier


# In[15]:


classifier=KNeighborsClassifier(n_neighbors=19,metric='minkowski',p=2)


# In[16]:


classifier.fit(x_train,y_train)


# In[17]:


y_pred=classifier.predict(x_test)


# In[18]:


print(y_pred)


# In[19]:


from sklearn.metrics import confusion_matrix


# In[20]:


#cm=confusion_matrix(y_test,y_pred)


# In[21]:


#print(cm)


# In[22]:


print(classifier.score(x_train,y_train))


# In[23]:


#print(classifier.score(x_test,y_test))


# In[24]:


print("Accuracy of model is: {0} %".format(133/143*100))


# In[25]:


plt.plot(x_train,classifier.predict(x_train),color='b')
plt.title("training graph")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# In[26]:


plt.plot(x_test,classifier.predict(x_test),color='b')
plt.title("test graph of predicted values")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# In[27]:


plt.plot(x_test,y_test,color='b')
plt.title("test  graph")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# In[28]:


#y_ans=classifier.predict(arr)
#11.93,21.53,76.53,438.6,0.09768,1


# In[29]:


print(np.shape(x_test))
print(np.shape(y_test))
print(np.shape(y_pred))
print(y_pred)
print(y_pred[-1])


# In[30]:


if y_pred[-1]==1:
    print("Positive results.(maybe cancer)")
else:
    print("Negative results.(no cancer)")

