
# coding: utf-8

# In[1]:


import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. 


# In[2]:


DATA_DIR = "data/fish_image"
IMG_SIZE = 100
LR = 1e-4

MODEL_NAME = 'fishCNN-{}.model'.format('conv-basic')

def createData():
    data=[]
    for dir in tqdm(os.listdir(DATA_DIR)):
        img_path = os.path.join(DATA_DIR,dir)
        label=dir.split('_')[-1];
        oneHotLabel=np.zeros(23);
        oneHotLabel[int(label)-1]=1;
        for img in tqdm(os.listdir(img_path)):
            path = os.path.join(img_path,img)
            img = cv2.imread(path)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            data.append([np.array(img),np.array(oneHotLabel)])
    shuffle(data)
    np.save('data.npy', data)
    return data


# In[ ]:





# In[3]:


#data=createData()
data=np.load("data.npy")


# In[4]:


data[1][0].size


# In[5]:


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 23, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


# In[6]:


if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')


# In[7]:


train = data[:-2000]
test = data[-2000:]


# In[8]:


X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
test_y = [i[1] for i in test]

#X.dtype

# X=X.view('float32')
# test_x/=test_x.view('float32')

# test_x
test_x.size


# In[10]:


model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)
