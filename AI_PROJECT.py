#BREAST CANCER PREDICTION MODEL 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#TAKING INPUT FROM THE USER IN ARRAY 
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
"""print(arr)"""

#LOADING DATA SET AND TAKING X-ARRAY AND Y-ARRAY
data=pd.read_csv('Breast_cancer_data.csv')
X=data.iloc[:,0:5].values
Y=data.iloc[:,5].values

#SPLITTING THE DATA SET INTO TRAIN AND TEST
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
x_test=np.append(x_test,arr,axis=0)

#SCALING THE DATA SET
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

# K NEAREST NEIGHBORS MODEL 
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=19,metric='minkowski',p=2)
classifier.fit(x_train,y_train)

#PREDICTING THE VALUES
y_pred=classifier.predict(x_test)

#CHECKING THE ACCURACY OF THE MODEL
print("accuracy of model is ",end="")
print((classifier.score(x_train,y_train))*100)

#PLOTTING THE GRAPH FOR TRAINING SET
"""plt.plot(x_train,classifier.predict(x_train),color='b')
plt.title("training graph")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()"""

#PLOTTING THE GRAPH FOR PREDICTION VALUES
"""plt.plot(x_test,classifier.predict(x_test),color='b')
plt.title("test graph of predicted values")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()"""

#OUTPUT FOR THE USER'S INPUT 
"""print(np.shape(x_test))
print(np.shape(y_test))
print(np.shape(y_pred))
print(y_pred)"""
print(y_pred[-1])

#PREDICTING IF CANCER OR NOT!
if y_pred[-1]==1:
    print("Positive results.(maybe cancer)")
else:
    print("Negative results.(no cancer)")
    