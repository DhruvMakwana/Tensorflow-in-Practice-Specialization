#!/usr/bin/env python
# coding: utf-8

# ## Exercise 3
# In the videos you looked at how you would improve Fashion MNIST using Convolutions. For your exercise see if you can improve MNIST to 99.8% accuracy or more using only a single convolutional layer and a single MaxPooling 2D. You should stop training once the accuracy goes above this amount. It should happen in less than 20 epochs, so it's ok to hard code the number of epochs for training, but your training must end once it hits the above metric. If it doesn't, then you'll need to redesign your layers.
# 
# I've started the code for you -- you need to finish it!
# 
# When 99.8% accuracy has been hit, you should print out the string "Reached 99.8% accuracy so cancelling training!"
# 

# In[5]:


import tensorflow as tf
print(tf.__version__)

# YOUR CODE STARTS HERE
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc') > .998):
            print("\nReached 99.8% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback() 
# YOUR CODE ENDS HERE

mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# check training and test image size
print("training image shape:{0} test image shape:{1}".format(training_images.shape,test_images.shape))

# YOUR CODE STARTS HERE
training_images = training_images.reshape(training_images.shape[0],training_images.shape[1],training_images.shape[2],1)
training_images = training_images/255.0
test_images = test_images.reshape(test_images.shape[0],test_images.shape[1],test_images.shape[2],1)
test_images = test_images/255.0
# YOUR CODE ENDS HERE

model = tf.keras.models.Sequential()
# YOUR CODE STARTS HERE
model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(524,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))
          
# YOUR CODE ENDS HERE

# YOUR CODE STARTS HERE
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(training_images,training_labels,epochs=20,callbacks=[callbacks])
# YOUR CODE ENDS HERE

test_loss = model.evaluate(test_images, test_labels)


# In[ ]:




