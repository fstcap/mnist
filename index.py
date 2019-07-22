import argparse
parser = argparse.ArgumentParser(description="manual to this script")
parser.add_argument('--models',type=str,default='cnn')
args = parser.parse_args()
models = args.models

print('models',models)

import numpy as np #linear algebra
import pandas as pd #data processing,CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

from keras.models import Sequential,load_model
from keras.layers import Dense ,Conv2D ,MaxPooling2D ,BatchNormalization , Dropout  , Lambda , Flatten
from keras.optimizers import Adam , RMSprop
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

#Input data files are available in the "../input/" directory
#For example , running this (by clicking run or pressing shift+Enter) will list the files in theinput directory

from subprocess import check_output
print(check_output(["ls","../input"]).decode("utf8"))

#Any result you write to the current directory are saved as output

#create the training & test sets,skipping the header row with[1:]
train = pd.read_csv("../input/train.csv")
print("train",train.shape)
print("train_5",train.head())
test = pd.read_csv("../input/test.csv")
print("test",test.shape)
print("test_5",test.head())

#convert train datset to (num_images,img_rows,img_cols) format
x_train = train.iloc[:,1:].values.astype('float32') #all pixel values
y_train = train.iloc[:,0].values.astype('int32') #only labels i.e targets digits
x_test = test.values.astype('float32')

x_train = x_train.reshape(x_train.shape[0],28,28)

for i in range(6,9):
	plt.subplot(330 + (i+1))
	plt.imshow(x_train[i],cmap=plt.get_cmap('gray'))
	plt.title(y_train[i])
#plt.show()
x_train = x_train.reshape(x_train.shape[0],28,28,1)
print("x_train",x_train.shape)
x_test = x_test.reshape(x_test.shape[0],28,28,1)
print("x_test",x_test.shape)

#It is important preprocessing step.It is used to centre the data around zero mean and unit variance.

def standardize(x_train_val_test):
	mean_px = x_train_val_test.mean().astype(np.float32)
	std_px = x_train_val_test.std().astype(np.float32)
	return (x_train_val_test-mean_px)/std_px

def lambda_model(x):
	return x

#One Hot encoding of labels
from keras.utils import to_categorical
y_train = to_categorical(y_train)
num_classes = y_train.shape[1]

#fix random seed for reproducibility
seed = 43
np.random.seed(seed)

#Linear Model
def get_fc_model():
	model = Sequential()
	model.add(Lambda(lambda_model,input_shape=(28,28,1)))
	model.add(Flatten())
	model.add(Dense(512,activation='relu'))
	model.add(Dense(10, activation='softmax'))
	model.compile(optimizer=RMSprop(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
	return model

#Convolutional Neural Network
def get_cnn_model():
	model = Sequential([
		Lambda(lambda_model,input_shape=(28,28,1)),
		Conv2D(filters=32,kernel_size=3,strides=1,padding='SAME',activation='relu'),
		MaxPooling2D(pool_size=2,padding='SAME'),
		BatchNormalization(),
		Conv2D(filters=64,kernel_size=3,strides=1,padding='SAME',activation='relu'),
		MaxPooling2D(pool_size=2,padding='SAME'),
		BatchNormalization(),
		Flatten(),
		Dense(512,activation='relu'),
		BatchNormalization(),
		Dense(10,activation='softmax')
	])
	model.compile(optimizer=Adam(lr=0.01),loss='categorical_crossentropy',metrics=['accuracy'])
	return model

#Cross Validation
x = x_train
y = y_train
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.10,random_state=seed)

x_train = standardize(x_train)
x_val = standardize(x_val)
x_test = standardize(x_test)
print('x_train',x_train.shape,'x_val',x_val.shape,'x_test',x_test.shape)

gen = ImageDataGenerator()
batches = gen.flow(x_train, y_train, batch_size=64)
val_batches = gen.flow(x_val, y_val, batch_size=64)
print("batches",batches.n)
print("val_batches",val_batches.n)

if models == 'fc':
	model = get_fc_model()
	filepath = 'logs/weights_fc.best.h5'
else:
	model = get_cnn_model()
	filepath = 'logs/weights_cnn.best.h5'

checkpoint = ModelCheckpoint(filepath,monitor='val_loss', save_weights_only=False, save_best_only=True, period=3,mode='auto')

history = model.fit_generator(generator=batches,steps_per_epoch=batches.n,epochs=1,validation_data=val_batches,validation_steps=val_batches.n)

history_dict = history.history
print(history_dict.keys())


#model_h5 = load_model(filepath)
predictions_h5 = model.predict_classes(x_test,verbose=0)

submissions = pd.DataFrame({'ImageId':list(range(1,len(predictions_h5)+1)),'Label':predictions_h5})
submissions.to_csv("DR.csv",index=False,header=True)
