
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten


# In[2]:


classifier=Sequential()


# In[3]:


classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation="relu"))


# In[4]:


classifier.add(MaxPooling2D(pool_size=(2,2)))


# In[5]:


classifier.add(Convolution2D(32,3,3,activation="relu"))


# In[6]:


classifier.add(MaxPooling2D(pool_size=(2,2)))


# In[7]:


classifier.add(Flatten())


# In[8]:


classifier.add(Dense(output_dim=128,activation="relu"))


# In[9]:


classifier.add(Dense(output_dim=1,activation="sigmoid"))


# In[10]:


classifier.compile(optimizer="adam",metrics=["accuracy"],loss="binary_crossentropy")


# In[11]:


from keras.preprocessing.image import ImageDataGenerator


# In[12]:


train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)


# In[13]:


test_datagen=ImageDataGenerator(rescale=1./255)


# In[14]:


training_set=train_datagen.flow_from_directory("dataset/training_set",target_size=(64,64),batch_size=32,class_mode="binary")


# In[15]:


test_set=test_datagen.flow_from_directory("dataset/test_set",target_size=(64,64),batch_size=32,class_mode="binary")


# In[24]:


classifier.fit_generator(training_set,samples_per_epoch=8000,nb_epoch=25,validation_data=test_set,nb_val_samples=2000)


# In[26]:


print("accuracy is 88 %")


# Epoch 1/25
# 250/250 [==============================] - 609s 2s/step - loss: 0.6714 - acc: 0.5864 - val_loss: 0.6145 - val_acc: 0.6899
# Epoch 2/25
# 250/250 [==============================] - 538s 2s/step - loss: 0.5997 - acc: 0.6761 - val_loss: 0.5570 - val_acc: 0.7170
# Epoch 3/25
# 250/250 [==============================] - 545s 2s/step - loss: 0.5669 - acc: 0.7054 - val_loss: 0.7037 - val_acc: 0.6547
# Epoch 4/25
# 250/250 [==============================] - 539s 2s/step - loss: 0.5467 - acc: 0.7239 - val_loss: 0.5765 - val_acc: 0.7008
# Epoch 5/25
# 250/250 [==============================] - 525s 2s/step - loss: 0.5183 - acc: 0.7420 - val_loss: 0.5011 - val_acc: 0.7560
# Epoch 6/25
# 250/250 [==============================] - 500s 2s/step - loss: 0.4971 - acc: 0.7543 - val_loss: 0.4967 - val_acc: 0.7618
# Epoch 7/25
# 250/250 [==============================] - 541s 2s/step - loss: 0.4825 - acc: 0.7676 - val_loss: 0.4892 - val_acc: 0.7606
# Epoch 8/25
# 250/250 [==============================] - 620s 2s/step - loss: 0.4649 - acc: 0.7761 - val_loss: 0.4769 - val_acc: 0.7660
# Epoch 9/25
# 250/250 [==============================] - 549s 2s/step - loss: 0.4514 - acc: 0.7928 - val_loss: 0.5218 - val_acc: 0.7575
# Epoch 10/25
# 250/250 [==============================] - 471s 2s/step - loss: 0.4480 - acc: 0.7866 - val_loss: 0.4992 - val_acc: 0.7574
# Epoch 11/25
# 250/250 [==============================] - 423s 2s/step - loss: 0.4300 - acc: 0.7991 - val_loss: 0.4977 - val_acc: 0.7617
# Epoch 12/25
# 250/250 [==============================] - 419s 2s/step - loss: 0.4118 - acc: 0.8059 - val_loss: 0.5040 - val_acc: 0.7703
# Epoch 13/25
# 250/250 [==============================] - 414s 2s/step - loss: 0.4010 - acc: 0.8199 - val_loss: 0.4839 - val_acc: 0.7847
# Epoch 14/25
# 250/250 [==============================] - 438s 2s/step - loss: 0.3947 - acc: 0.8223 - val_loss: 0.5146 - val_acc: 0.7728
# Epoch 15/25
# 250/250 [==============================] - 445s 2s/step - loss: 0.3764 - acc: 0.8288 - val_loss: 0.4888 - val_acc: 0.7897
# Epoch 16/25
# 250/250 [==============================] - 446s 2s/step - loss: 0.3642 - acc: 0.8309 - val_loss: 0.4731 - val_acc: 0.7790
# Epoch 17/25
# 250/250 [==============================] - 459s 2s/step - loss: 0.3530 - acc: 0.8408 - val_loss: 0.4767 - val_acc: 0.7927
# Epoch 18/25
# 250/250 [==============================] - 574s 2s/step - loss: 0.3513 - acc: 0.8399 - val_loss: 0.4747 - val_acc: 0.7908
# Epoch 19/25
# 250/250 [==============================] - 871s 3s/step - loss: 0.3384 - acc: 0.8505 - val_loss: 0.4851 - val_acc: 0.8005
# Epoch 20/25
# 250/250 [==============================] - 932s 4s/step - loss: 0.3273 - acc: 0.8530 - val_loss: 0.4807 - val_acc: 0.7995
# Epoch 21/25
# 250/250 [==============================] - 1052s 4s/step - loss: 0.3201 - acc: 0.8592 - val_loss: 0.4848 - val_acc: 0.7926
# Epoch 22/25
# 250/250 [==============================] - 938s 4s/step - loss: 0.3040 - acc: 0.8656 - val_loss: 0.4968 - val_acc: 0.7947
# Epoch 23/25
# 250/250 [==============================] - 931s 4s/step - loss: 0.3006 - acc: 0.8723 - val_loss: 0.5222 - val_acc: 0.7978
# Epoch 24/25
# 250/250 [==============================] - 978s 4s/step - loss: 0.2933 - acc: 0.8719 - val_loss: 0.5397 - val_acc: 0.7910
# Epoch 25/25
# 250/250 [==============================] - 514s 2s/step - loss: 0.2768 - acc: 0.8810 - val_loss: 0.5702 - val_acc: 0.7757
# <keras.callbacks.History at 0x11969b6d4a8>
