#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import cv2


# In[26]:


with_mask = np.load('with_mask.npy')
without_mask = np.load('without_mask.npy')


# In[27]:


with_mask.shape


# In[28]:


without_mask.shape


# In[29]:


#converting 3 dimensional data into 2 dimensional data
with_mask = with_mask.reshape(1000,50 * 50 * 3)
without_mask = without_mask.reshape(1000,50 * 50 * 3)


# In[30]:


#converted
with_mask.shape


# In[31]:


without_mask.shape


# In[32]:


#combined those arrays together and stored it in X
X = np.r_[with_mask, without_mask]     #r_ is used to link toghether the rows


# In[33]:


X.shape


# In[34]:


labels = np.zeros(X.shape[0]) #y


# In[35]:


# after 400 images the value will be one (means with mask)
labels[1000:] = 1.0


# In[36]:


names = {0: 'Mask', 1: 'No Mask'}


# # ML AGORITHM

# In[37]:


# SVM --> support vector machine
# SVC --> support vector classification
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score


# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size = 0.28)


# In[40]:


x_train.shape


# In[41]:


# we have so many columns that can slow down the processing
# we have to use the dimensionality reduction technique


# In[42]:


#using package decompsition and importing PCA (principle component analysis)
from sklearn.decomposition import PCA


# In[43]:


pca = PCA(n_components = 3)
x_train = pca.fit_transform(x_train)


# In[44]:


x_train[0]


# In[45]:


#Reduced the columns
x_train.shape


# In[46]:


svm = SVC()
svm.fit(x_train, y_train)


# In[47]:


x_test = pca.transform(x_test)
y_pred = svm.predict(x_test)


# In[48]:


accuracy_score(y_test, y_pred)


# In[49]:


haar_data = cv2.CascadeClassifier('data.xml')
capture = cv2.VideoCapture(0) 
data = []
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    flag, img = capture.read() 
    if flag:
        faces = haar_data.detectMultiScale(img) 
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 2)
            face = img[y:y+h, x:x+h, :] # slicing the face from image
            face = cv2.resize(face,(50,50)) # Resizing all the images dimensions
            face = face.reshape(1,-1)
            face = pca.transform(face)
            pred = svm.predict(face)[0]
            n = names[int(pred)]
            cv2.putText(img, n, (x,y), font, 1, (244,250,250), 4)
            print(n)
        cv2.imshow('result',img)
        if cv2.waitKey(2) == 27: 
            break
capture.release() 
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




