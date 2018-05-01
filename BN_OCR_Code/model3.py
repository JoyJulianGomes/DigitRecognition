#Required Imports----------------------------------------------------------------------------------
import numpy as np
from scipy.io import loadmat
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import utils, optimizers
from matplotlib import pyplot as plt
#--------------------------------------------------------------------------------------------------


#Data Loading--------------------------------------------------------------------------------------

data = loadmat('E:\Work\Projects\DigitRecognition\Bengali_Dataset\BN_NUM_CHARS.mat')
testX = data['testX']
testY = utils.to_categorical(data['testY'], num_classes = 10)
trainX = data['trainX']
trainY = utils.to_categorical(data['trainY'], num_classes = 10)

print("Training Set Images - {} instances of length {}".format(trainX.shape[0], trainX.shape[1]))
print("Training Set Labels - {} instances of length {}".format(trainY.shape[0], trainY.shape[1]))
print("Testing Set Images - {} instances of length {}".format(testX.shape[0], testX.shape[1]))
print("Testing Set Labels - {} instances of length {}".format(testY.shape[0], testY.shape[1]))
#--------------------------------------------------------------------------------------------------


#Designing Models----------------------------------------------------------------------------------

#Model Small Equal:
#image input = 400
#image output = 10
#Difference = (400-10)/2=195
#Inp-H1-Out
#400-195-10

#Model Small Exponential:
#image input = 400
#image output = 10
#Difference = (400-10)/3=130
#Inp-H1-|-Out
#400-130-|-10

model_1_S_Eq = Sequential()
model_1_S_Eq.add(Dense(195, input_shape=(400,)))
model_1_S_Eq.add(Activation('sigmoid'))
model_1_S_Eq.add(Dense(10))
model_1_S_Eq.add(Activation('sigmoid'))

model_2_S_Ex = Sequential()
model_2_S_Ex.add(Dense(130, input_shape=(400,)))
model_2_S_Ex.add(Activation('sigmoid'))
model_2_S_Ex.add(Dense(10))
model_2_S_Ex.add(Activation('sigmoid'))

'''
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model_1_S_Eq.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
'''

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model_2_S_Ex.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
#---------------------------------------------------------------------------------------------------


#Training and Evaluating The Models-----------------------------------------------------------------
'''
history = model_1_S_Eq.fit(trainX, trainY, epochs=40, validation_split=0.20, batch_size=10)
model_1_S_Eq_Score = model_1_S_Eq.evaluate(testX, testY)
print("Maximum Validation Accuracy: {}\nMaximum Validation Epoch: {}".format(np.amax(history.history['val_acc']), np.argmax(history.history['val_acc'])+1))
model_1_S_Eq.save('model_1_S_Eq.h5')
'''

history = model_2_S_Ex.fit(trainX, trainY, epochs=31, validation_split=0.20, batch_size=10)
model_1_S_Eq_Score = model_2_S_Ex.evaluate(testX, testY)
print("Maximum Validation Accuracy: {}\nMaximum Validation Epoch: {}".format(np.amax(history.history['val_acc']), np.argmax(history.history['val_acc'])+1))
model_2_S_Ex.save('model_2_S_Ex.h5')


# list all data in history
#print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#---------------------------------------------------------------------------------------------------