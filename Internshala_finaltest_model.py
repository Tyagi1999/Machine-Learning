
# coding: utf-8

# In[62]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[63]:


#training dataframe
train_data=pd.read_csv("Internshala_train.csv")


# In[64]:


train_data.tail()


# In[65]:


#checking for null values
train_data.isnull().sum()


# In[66]:


train_data.describe()


# In[67]:


train_data.shape


# In[68]:


#counting the terms of 'no' and 'yes'
train_data['subscribed'].value_counts()


# In[69]:


train_data.plot.box()


# In[70]:


#changing categorical values to numerical
train_data['subscribed'].replace('no',0,inplace=True)
train_data['subscribed'].replace('yes',1,inplace=True)


# In[71]:


#getting dummies for categorical values
train_data=pd.get_dummies(train_data)


# In[72]:


#getting input values
x=train_data.drop('subscribed',1)


# In[73]:


#getting output values
y=train_data['subscribed']


# In[74]:


#scaling the training values
from sklearn.preprocessing import StandardScaler


# In[75]:


sc=StandardScaler()


# In[76]:


x=sc.fit_transform(x)


# In[77]:


#logistic regression
from sklearn.linear_model import LogisticRegression


# In[78]:


regressor=LogisticRegression()


# In[79]:


#fitting the values
regressor.fit(x,y)


# In[80]:


#accuracy of training
print(regressor.score(x,y))


# In[81]:


#test data
test_data=pd.read_csv("Internshala_test.csv")


# In[82]:


#checking null values
test_data.isnull().sum()


# In[83]:


#getting dummies
test_data=pd.get_dummies(test_data)


# In[84]:


#getting input
x_test=test_data.iloc[:,0:]


# In[85]:


#scaling testing values
x_test=sc.transform(x_test)


# In[86]:


#predicting values for test data(**its accuracy is above 90%**)
y_pred=regressor.predict(x_test)


# In[87]:


#adding result column to dataframe
test_data['subscribed']=y_pred


# In[88]:


test_data.head()


# In[89]:


#converting resule from numerical to categorical values
test_data['subscribed'].replace(0,'no',inplace=True)
test_data['subscribed'].replace(1,'yes',inplace=True)


# In[90]:


test_data.head()


# In[91]:


#taking last column from dataframe
z=test_data.iloc[:,-1]


# In[92]:


#converting this last column to csv file
z.to_csv('pred_test.csv',header=True,index=False)


# In[93]:


#Decision tree regression
from sklearn.tree import DecisionTreeClassifier


# In[94]:


tree_reg=DecisionTreeClassifier(max_depth=4,random_state=0)


# In[95]:


#fitting values
tree_reg.fit(x,y)


# In[96]:


#trainig score
print(tree_reg.score(x,y))


# In[97]:


#predicting values
y_pred_2=tree_reg.predict(x_test)


# In[98]:


y_pred_2


# In[99]:


test_data_2=pd.read_csv('Internshala_test.csv')


# In[100]:


test_data_2.head()


# In[101]:


test_data_2['subscribed']=y_pred_2


# In[102]:


test_data_2.head()


# In[103]:


test_data_2['subscribed'].replace(0,'no',inplace=True)
test_data_2['subscribed'].replace(1,'yes',inplace=True)


# In[104]:


z_2=test_data_2.iloc[:,-1]


# In[105]:


z_2.to_csv('pred_test_2.csv',header=True,index=False)


# In[106]:


#accuracy of both the models
print("Accuracy of LOgistic regression = {0}".format(90.6))
print("Accuracy of decision tree classifier = {0}".format(90.9))


# In[107]:


#error metrix for logistic regression
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y,regressor.predict(x))
print(cm)


# In[108]:


#error metrix for decision tree classifier
cm_2=confusion_matrix(y,tree_reg.predict(x))
print(cm_2)

