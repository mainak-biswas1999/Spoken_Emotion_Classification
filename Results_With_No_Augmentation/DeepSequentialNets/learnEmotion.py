import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

ACCURACY_THRESHOLD = 0.995

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') >= ACCURACY_THRESHOLD):
            print("\nReached %2.2f%% accuracy, so stopping training!!" %(ACCURACY_THRESHOLD*100))
            self.model.stop_training = True

def printMetrics(model_LSTM, Xtrain, Ytrain, Xtest, Ytest, printdetails):
    print("sum of the columns give number of examples per class, i.e A[i][j] = A[i][j]+1 if item in jth class is classified to be in the ith class", file=printdetails)
    print("Class are in Order: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised", file=printdetails)
    
    #confusion matrix, Train
    predictions_train = model_LSTM.predict(Xtrain)
    predtrain_y = predictions_train.argmax(axis=1) 
    conf_matTrain = np.zeros((8, 8))
    for i in range(predtrain_y.shape[0]):
        conf_matTrain[predtrain_y[i], Ytrain[i].astype(np.int8)] = conf_matTrain[predtrain_y[i], Ytrain[i].astype(np.int8)] + 1
    
    print("Train Accuracy = ",'{0:.3g}'.format(conf_matTrain.trace()*100/predtrain_y.shape[0]),"%", file=printdetails)
    print(conf_matTrain, file=printdetails)
    
    #Confusion Matrix, Test
    predictions_test = model_LSTM.predict(Xtest)
    predtest_y = predictions_test.argmax(axis=1)
    conf_matTest = np.zeros((8, 8))
    for i in range(predtest_y.shape[0]):
        conf_matTest[predtest_y[i], Ytest[i].astype(np.int8)] = conf_matTest[predtest_y[i], Ytest[i].astype(np.int8)] + 1
    
    print("Test Accuracy = ",'{0:.3g}'.format(conf_matTest.trace()*100/predtest_y.shape[0]),"%", file=printdetails)
    print(conf_matTest, file=printdetails)    
    

def plotfeatures(history, direc):
    #accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(direc+"accuracy.png")
    #loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(direc+"loss.png")
    
def train_test_LSTM(Xtrain_loc, Ytrain_loc, Xtest_loc, Ytest_loc):
    #for y accepts only 0-n-1
    printdetails = open("ModelPerformance/Results.txt", 'w')
    Xtrain = np.load(Xtrain_loc)
    Ytrain = np.load(Ytrain_loc) - 1
    Xtest = np.load(Xtest_loc)
    Ytest = np.load(Ytest_loc) - 1
    print("Shape of X: ",Xtrain.shape, Ytrain.shape)
    print("Shape of Y", Xtest.shape, Ytest.shape)
    
    #load/create model
    model_LSTM = Sequential()
    #model_LSTM = load_model("Model/LSTM_Model_forEmotion")
    
    #model_LSTM.add(LSTM(256, input_shape=(Xtrain.shape[1:]), activation='relu', return_sequences=False))
    model_LSTM.add(LSTM(32, input_shape=(Xtrain.shape[1:]), return_sequences=False))    #tanh
    model_LSTM.add(Dropout(0.10))    #dropping out due to the large output size and for regularization
    
    #model_LSTM.add(LSTM(128, activation='relu', return_sequences=False))
    model_LSTM.add(Dense(8, activation='softmax'))
    
    optimization_obj = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-5)
    model_LSTM.compile(loss='sparse_categorical_crossentropy', optimizer=optimization_obj, metrics = ['accuracy'])
    model_LSTM.summary()

    # Instantiate a callback object
    callbacks = myCallback()

    #Backprop
    history = model_LSTM.fit(Xtrain, Ytrain, epochs = 2000, batch_size=64, validation_data = (Xtest, Ytest), callbacks=[callbacks])
    
    #save
    model_LSTM.save("Model/LSTM_Model_forEmotion")
    
    #print the results 
    printMetrics(model_LSTM, Xtrain, Ytrain, Xtest, Ytest, printdetails)
    plotfeatures(history, "ModelPerformance/")
    printdetails.close()
    
    
def train_test_GRU(Xtrain_loc, Ytrain_loc, Xtest_loc, Ytest_loc):
    #for y accepts only 0-n-1
    printdetails = open("ModelPerformance/ResultsGRU.txt", 'w')
    Xtrain = np.load(Xtrain_loc)
    Ytrain = np.load(Ytrain_loc) - 1
    Xtest = np.load(Xtest_loc)
    Ytest = np.load(Ytest_loc) - 1
    print("Shape of X: ",Xtrain.shape, Ytrain.shape)
    print("Shape of Y", Xtest.shape, Ytest.shape)
    
    #load/create model
    model_GRU = Sequential()
    #model_LSTM = load_model("Model/LSTM_Model_forEmotion")
    
    #model_LSTM.add(LSTM(256, input_shape=(Xtrain.shape[1:]), activation='relu', return_sequences=False))
    model_GRU.add(GRU(32, input_shape=(Xtrain.shape[1:]), return_sequences=False))    #tanh
    model_GRU.add(Dropout(0.5))    #dropping out due to the large output size and for regularization
    
    #model_LSTM.add(LSTM(128, activation='relu', return_sequences=False))
    model_GRU.add(Dense(8, activation='softmax'))
    
    optimization_obj = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-5)
    model_GRU.compile(loss='sparse_categorical_crossentropy', optimizer=optimization_obj, metrics = ['accuracy'])
    model_GRU.summary()

    # Instantiate a callback object
    callbacks = myCallback()

    #Backprop
    history = model_GRU.fit(Xtrain, Ytrain, epochs = 2000, batch_size=64, validation_data = (Xtest, Ytest), callbacks=[callbacks])
    
    #save
    model_GRU.save("Model/GRU_Model_forEmotion")
    
    #print the results 
    printMetrics(model_GRU, Xtrain, Ytrain, Xtest, Ytest, printdetails)
    plotfeatures(history, "ModelPerformance/")
    printdetails.close()
    
def train_test_CNN_logSpect(Xtrain_loc, Ytrain_loc, Xtest_loc, Ytest_loc):
    #for y accepts only 0-n-1
    printdetails = open("ModelPerformance/Results_CNN_logSpect.txt", 'w')
    #norm_Max = 80
    Xtrain = np.load(Xtrain_loc)/80
    Ytrain = np.load(Ytrain_loc) - 1
    Xtest = np.load(Xtest_loc)/80
    Ytest = np.load(Ytest_loc) - 1
    
    #converting to 4D data
    Xtrain = Xtrain.reshape((-1, Xtrain.shape[1], Xtrain.shape[2], 1))
    Xtest = Xtest.reshape((-1, Xtest.shape[1], Xtest.shape[2], 1))
    
    print("Shape of X: ",Xtrain.shape, Ytrain.shape)
    print("Shape of Y", Xtest.shape, Ytest.shape)
    #load/create model
    model_CNN = Sequential()
    #model_LSTM = load_model("Model/CNN_logSpect_Model_forEmotion")
    
    #layer 1(Conv): 1025x160x1 ----> 205x52x8
    model_CNN.add(Conv2D(16, kernel_size=(5,5), strides = (5,3), input_shape=(Xtrain.shape[1:]), activation='relu'))    
    
    #layer 2(MaxPool): 205x52x8 ----> 102x26x8
    model_CNN.add(MaxPooling2D(pool_size=(2,2), strides= (2,2)))

    #layer 3: 102x26x8 ----> 34x12x16
    model_CNN.add(Conv2D(32, kernel_size=(3,3), strides = (3,2), activation='relu'))
    
    #layer 4(MaxPool): 34x12x16 ----> 17x6x16
    model_CNN.add(MaxPooling2D(pool_size=(2,2), strides= (2,2)))
    
    #layer 5(Deep): 1632  ---> 128 
    model_CNN.add(Flatten())
    model_CNN.add(Dropout(0.2))
    model_CNN.add(Dense(256, activation='relu'))
    
    #layer 6(output)
    model_CNN.add(Dropout(0.2))
    model_CNN.add(Dense(8, activation='softmax'))
    
    optimization_obj = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-5)
    model_CNN.compile(loss='sparse_categorical_crossentropy', optimizer=optimization_obj, metrics = ['accuracy'])
    model_CNN.summary()

    # Instantiate a callback object
    callbacks = myCallback()

    #Backprop
    history = model_CNN.fit(Xtrain, Ytrain, epochs = 2000, batch_size=64, validation_data = (Xtest, Ytest), callbacks=[callbacks])
    
    #save
    model_CNN.save("Model/CNN_logSpect_Model_forEmotion")
    
    #print the results 
    printMetrics(model_CNN, Xtrain, Ytrain, Xtest, Ytest, printdetails)
    plotfeatures(history, "ModelPerformance/")
    printdetails.close()
    
