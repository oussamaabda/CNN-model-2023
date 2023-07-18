# CNN-model-2023
import numpy as np  # numpy library
np.random.seed(123) #creating aleatoire nuber

from keras.models import Sequential # sampling for CNN
from keras.layers import Dense, Dropout, Activation, Flatten #important layers for any models
from keras.layers import Convolution2D, MaxPooling2D # for cnn
from keras.utils import np_utils #  import tools for NN building

### now we have all tools and library for building CNN

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test)= mnist.load_data(path="mnist.npz")

print (X_train.shape)

from matplotlib import pyplot as plt
plt.imshow(X_train[0]) # ??

## preprocessing in put data

X_train = X_train.reshape(X_train.shape[0],28,28,1)  # change dimension
X_test = X_test.reshape(X_test.shape[0],28,28,1)  # change dimension (depth)
print(X_train.shape)
## the last steps of preprocessing is normalisation of pixels values
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255   ## between 0 and 1
X_test /= 255

# step 6 : class lebels

print(y_train.shape)

print(y_train[:10])


Y_train = np_utils.to_categorical(y_train,10)  # devide datasets into classes
Y_test = np_utils.to_categorical(y_test,10)
print(Y_train.shape)

## medel building (step 7)

model = Sequential()
model.add(Convolution2D(32, 3,3, activation ='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))

## step 8 :compile
model.compile(loss ='categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

## fit model step 9

model.fit(X_train, Y_train,
          batch_size = 32, nb_epoch = 10, verbose = 1)

## step 10 
score = model.evaluate(X_test, Y_test , verbose = 0)
