#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opencv-python')


# In[3]:


import cv2


# In[4]:


img = cv2.imread('img2.webp') # reading the image 


# In[5]:


img.shape #size of the sample image


# In[6]:


img[0]#array of first row of our image


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


plt.imshow(img)


# In[9]:


#This will show the images
while True:
    cv2.imshow('result',img)
    if cv2.waitKey(2) == 27: #Here 27 is the ASCII value of esc key, to stop the execution and 2 is the time in miliseconds
        break
cv2.destroyAllWindows()

BY VIOLA JONES ALGO:
    
This algo has four stages:
    1. Haar feature Selection - features that are present in human face, which are dark and white regions
    2. creating an integral image
    3. Adaboost Training
    4. Cascading Classifiers
# In[10]:


haar_data = cv2.CascadeClassifier('data.xml') #imported the haarcode feature file from github


# In[11]:


haar_data.detectMultiScale(img) #search for the haar features in every sliding window
#array of each image is return


# In[12]:


#cv2.rectangle(img,(x,y),(w,h),(b,g,r),border_thickness)


# In[13]:


# now add this detect scale into our image and display the image
while True:
    faces = haar_data.detectMultiScale(img)
    for x,y,w,h in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 2)
    cv2.imshow('result',img)
    if cv2.waitKey(2) == 27: 
        break
cv2.destroyAllWindows()


# In[14]:


capture = cv2.VideoCapture(0) #to open the camera

while True:
    flag, img = capture.read() #flag will tell if there is an issue, if FLAG== True then camera is working
    if flag:
        faces = haar_data.detectMultiScale(img) #will return faces from the image
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 2)
            
        cv2.imshow('result',img)
        if cv2.waitKey(2) == 27: 
            break
capture.release() #Used to release the camera that python holding
cv2.destroyAllWindows()


# In[25]:


# first run---> Taking the data from camera without mask
# second run --> Taking the data from camera with mask
capture = cv2.VideoCapture(0) 
data = []
while True:
    flag, img = capture.read() 
    if flag:
        faces = haar_data.detectMultiScale(img) 
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 2)
            face = img[y:y+h, x:x+h, :] # slicing the face from image
            face = cv2.resize(face,(50,50)) # Resizing all the images dimensions
            print(len(data)) 
            if len(data) < 1000: #taking 400 data
                data.append(face)
        cv2.imshow('result',img)
        if cv2.waitKey(2) == 27 or len(data) >= 1000: 
            break
capture.release() 
cv2.destroyAllWindows()


# In[16]:


import numpy as np


# In[26]:


np.save('without_mask.npy',data) #saving data after fisrt run


# In[19]:


np.save('with_mask.npy',data) #saving data after second run


# In[59]:


plt.imshow(data[532])


# In[22]:


plt.imshow(data[11])


# In[ ]:




