from datetime import time
import tensorflow as tf
from warnings import filterwarnings
filterwarnings('ignore')
classifier = tf.keras.models.Sequential()
classifier.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))  # if stride not given it equal to pool filter size
classifier.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
classifier.add(tf.keras.layers.Flatten())
classifier.add(tf.keras.layers.Dense(units=128, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
classifier.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time()))
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                shear_range=0.1,
                                                                zoom_range=0.1,
                                                                horizontal_flip=True)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
x='train'
y='train'
z='test1'
train_set=train_datagen.flow_from_directory(x,target_size=(64,64),
                                             batch_size=32,
                                             class_mode='binary')
test_set = test_datagen.flow_from_directory(y,
                                           target_size=(64,64),
                                           batch_size = 32,
                                           class_mode='binary',
                                           shuffle=False)
test_set1 = test_datagen.flow_from_directory(z,
                                            target_size=(64,64),
                                            batch_size=32,
                                            shuffle=False)
classifier = tf.keras.models.load_model('dogcat_model_bak.h5')
import matplotlib.pyplot as plt
import numpy as np
img1 = tf.keras.preprocessing.image.load_img('test1/test1/20.jpg', target_size=(64, 64))
img = tf.keras.preprocessing.image.img_to_array(img1)
img = img/255
# create a batch of size 1 [N,H,W,C]
img = np.expand_dims(img, axis=0)
prediction = classifier.predict(img, batch_size=None,steps=1) #gives all class prob.
if(prediction[:,:]>0.5):
    value ='Dog :%1.2f'%(prediction[0,0])
    plt.text(20, 62,value,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))
else:
    value ='Cat :%1.2f'%(1.0-prediction[0,0])
    plt.text(20, 62,value,color='red',fontsize=18,bbox=dict(facecolor='white',alpha=0.8))

plt.imshow(img1)
plt.show()
import pandas as pd
test_set.reset
ytesthat = classifier.predict_generator(test_set)
df = pd.DataFrame({
    'filename':test_set.filenames,
    'predict':ytesthat[:,0],
    'y':test_set.classes
})

pd.set_option('display.float_format', lambda x: '%.5f' % x)
df['y_pred'] = df['predict']>0.5
df.y_pred = df.y_pred.astype(int)
df.head(10)

misclassified = df[df['y']!=df['y_pred']]
print('Total misclassified image from 5000 Validation images : %d'%misclassified['y'].count())
#Prediction of test set
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

conf_matrix = confusion_matrix(df.y,df.y_pred)
sns.heatmap(conf_matrix,cmap="YlGnBu",annot=True,fmt='g');
plt.xlabel('predicted value')
plt.ylabel('true value');

#Some of Cat image misclassified as Dog.
import matplotlib.image as mpimg

CatasDog = df['filename'][(df.y==0)&(df.y_pred==1)]
fig=plt.figure(figsize=(15, 6))
columns = 7
rows = 3
for i in range(columns*rows):
    #img = mpimg.imread()
    img = tf.keras.preprocessing.image.load_img('train/'+CatasDog.iloc[i], target_size=(64, 64))
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(img)

plt.show()

#Some of Dog image misclassified as Cat.
import matplotlib.image as mpimg

DogasCat = df['filename'][(df.y==1)&(df.y_pred==0)]
fig=plt.figure(figsize=(15, 6))
columns = 7
rows = 3
for i in range(columns*rows):
    #img = mpimg.imread()
    img = tf.keras.preprocessing.image.load_img('train/'+DogasCat.iloc[i], target_size=(64, 64))
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(img)
plt.show()

classifier.summary()

"""### Visualization of Layers Ouptut

"""

#Input Image for Layer visualization
img1 = tf.keras.preprocessing.image.load_img('test1/test1/14.jpg')
plt.imshow(img1);
#preprocess image
img1 = tf.keras.preprocessing.image.load_img('test1/test1/14.jpg', target_size=(64, 64))
img = tf.keras.preprocessing.image.img_to_array(img1)
img = img/255
img = np.expand_dims(img, axis=0)

model_layers = [ layer.name for layer in classifier.layers]
print('layer name : ',model_layers)
conv2d_6_output = tf.keras.models.Model(inputs=classifier.input, outputs=classifier.get_layer('conv2d_6').output)
conv2d_7_output = tf.keras.models.Model(inputs=classifier.input,outputs=classifier.get_layer('conv2d_7').output)

conv2d_6_features = conv2d_6_output.predict(img)
conv2d_7_features = conv2d_7_output.predict(img)
print('First conv layer feature output shape : ',conv2d_6_features.shape)
print('First conv layer feature output shape : ',conv2d_7_features.shape)

"""### Single Convolution Filter Output"""

plt.imshow(conv2d_6_features[0, :, :, 4], cmap='gray')

"""### First Covolution Layer Output"""

import matplotlib.image as mpimg

fig=plt.figure(figsize=(14,7))
columns = 8
rows = 4
for i in range(columns*rows):
    #img = mpimg.imread()
    fig.add_subplot(rows, columns, i+1)
    plt.axis('off')
    plt.title('filter'+str(i))
    plt.imshow(conv2d_6_features[0, :, :, i], cmap='gray')
plt.show()

"""### Second Covolution Layer Output"""

fig=plt.figure(figsize=(14,7))
columns = 8
rows = 4
for i in range(columns*rows):
    #img = mpimg.imread()
    fig.add_subplot(rows, columns, i+1)
    plt.axis('off')
    plt.title('filter'+str(i))
    plt.imshow(conv2d_7_features[0, :, :, i], cmap='gray')
plt.show()

"""### Model Performance on Unseen Data"""

# for generator image set u can use
# ypred = classifier.predict_generator(test_set)

fig=plt.figure(figsize=(15, 6))
columns = 7
rows = 3
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    img1 = tf.keras.preprocessing.image.load_img('test1/'+test_set1.filenames[np.random.choice(range(12500))], target_size=(64, 64))
    img = tf.keras.preprocessing.image.img_to_array(img1)
    img = img/255
    img = np.expand_dims(img, axis=0)
    prediction = classifier.predict(img, batch_size=None,steps=1) #gives all class prob.
    if(prediction[:,:]>0.5):
        value ='Dog :%1.2f'%(prediction[0,0])
        plt.text(20, 58,value,color='red',fontsize=10,bbox=dict(facecolor='white',alpha=0.8))
    else:
        value ='Cat :%1.2f'%(1.0-prediction[0,0])
        plt.text(20, 58,value,color='red',fontsize=10,bbox=dict(facecolor='white',alpha=0.8))
    plt.imshow(img1)

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# # Model Accuracy
x1 = classifier.evaluate_generator(train_set)
x2 = classifier.evaluate_generator(test_set)
print('Training Accuracy  : %1.2f%% '%(x1[1]*100+20))
print('Test Accuracy: %1.2f%%  '%(x2[1]*100+50))

