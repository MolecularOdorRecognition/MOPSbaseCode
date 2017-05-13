"""
Step 2 :: NN Batch Size Variation for NN size 127 to 129
This code runs the NN alongwith creates the Necessary Visualizations for the
 created Model as well as logs the pertinent data to the epochs for Training and
 Validation, as well as the Testing Loss and Metrics.

BUG's:
    1. increased RAM usage in iPython console due to the scrolling of results. 
        On the other hand if the graphs are not displayed they simply overlay on
        top of each other  and become useless visualizations.

"""
import numpy
import keras
import pandas
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np

def printData(size, nBatch, history, fileName="results/results.csv", learnR=0.01, drop=0.1):
    """
    This function gets passed all values to be passed onto the CSV file.
    The default value of the file path is set to the directory in which this
    code is saved.
    The required function parameters, size, nBatch and history contain the data
    to be pushed to the file, however the data such as LR do not change
    throughout these codes.
    
    Data being saved contains the Max, the Min and Average of the metrics and 
    losses in order to get the prescribed information out of it - Most models 
    are considered on their Avg Metrics, however certain metrics may include 
    Max and even Min of the Metrics/Loss during training/validation.
    """
    out = open(fileName, "a")
    out.write("\n")
    out.write(str(size))
    out.write(",")
    out.write(str(nBatch))
    out.write(",")
    out.write(str(learnR))
    out.write(",")
    out.write(str(drop))
    out.write(",")
    out.write(str(max(history.history['loss'])))
    out.write(",")
    out.write(str(min(history.history['loss'])))
    out.write(",")
    out.write(str(sum(history.history['loss'])/len((history.history['loss']))))
    out.write(",")
    out.write(str(max(history.history['pearson_correlation'])))
    out.write(",")
    out.write(str(min(history.history['pearson_correlation'])))
    out.write(",")
    out.write(str(sum(history.history['pearson_correlation'])/len((history.history['pearson_correlation']))))
    out.write(",")
    out.write(str(max(history.history['val_loss'])))
    out.write(",")
    out.write(str(min(history.history['val_loss'])))
    out.write(",")
    out.write(str(sum(history.history['val_loss'])/len((history.history['val_loss']))))
    out.write(",")
    out.write(str(max(history.history['val_pearson_correlation'])))
    out.write(",")
    out.write(str(min(history.history['val_pearson_correlation'])))
    out.write(",")
    out.write(str(sum(history.history['val_pearson_correlation'])/len((history.history['val_pearson_correlation']))))
    out.flush()
    out.close()

#input data
X = pandas.read_csv("X.csv", header=None, low_memory=False, skiprows=1)
Y = pandas.read_csv("Y.csv", header=None, low_memory=False, skiprows=1)
print("Width of Input Layer:: ", len(X[0]))
X_train = np.asarray(X[:267])
Y_train = np.asarray(Y[:267])
X_test = np.asarray(X[267:])
Y_test = np.asarray(Y[267:])
del(X)
del(Y)
# deletion of data X, Y frees up valuable space

#HYPERPARAMETERS
numpy.random.seed(7)
nEpoch = 25
nBatch = 32
drop = 0.1
activName = 'hard_sigmoid'
initializerName = 'normal'
learn = 0.01
size = 64
loss = 'mean_absolute_error'

# default values to be placed in Logger File


# model creation, visualization creation of particular models and logging
for size in range(127, 130):
    out = open("results/"+str(size)+"results.csv", "a")
    out.write("\n")
    out.write("Neuron Number Hidden Layer 1,nBatch,drop,learning Rate,Max Training Loss, Min Training Loss, Avg Training Loss, ")
    out.write("Max Correlation Training, Min Correlation Training, Avg Correlation Training,")
    out.write("Max Validation Loss, Min Validation Loss, Avg Validation Loss,")
    out.write("Max Correlation Validation, Min Correlation Validation, Avg Correlation Validation,")
    out.flush()
    out.close()
    for nBatch in range(1, 65):
        print("Hidden Layer 1 Size", size, "nBatch ", nBatch, "learning rate=", learn, "drop ", drop)
        # model creation
        model = Sequential()
        model.add(Dense(size, input_dim=len(X_train[0]), kernel_initializer=initializerName, activation=activName))
        model.add(Dropout(drop))
        model.add(Dense(1, kernel_initializer=initializerName))
    
        # compiling model
        optim = optimizers.adam(lr=learn)
        model.compile(loss=loss, metrics=['pearson_correlation'], optimizer=optim)
    
        # callbacks
        redLRplaateau = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, verbose=1, cooldown=1)
        loggerFileName = "csvLogger/Size"+str(size)+"_B"+str(nBatch)+"_L"+str(learn)+"_D"+str(drop)+"_results.csv"
        csvLogger = keras.callbacks.CSVLogger(loggerFileName, append=True, separator=",")
    
        # training, validating
        history = model.fit(X_train, Y_train, batch_size=nBatch, epochs=nEpoch, verbose=0, validation_split=0.25, shuffle=True, callbacks=[redLRplaateau, csvLogger])

        # logging of data
        printData(size=size, nBatch=nBatch,history=history, learnR=learn, drop=drop,fileName="results/"+str(size)+"results.csv") #"+str(size)+"
        
        # graphical representation of history callback
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['Training', 'Validation'], loc='upper left')
        plt.savefig('report/loss_'+str(size)+'_B'+str(nBatch)+'_L'+str(learn)+'_D'+str(drop)+'.png')
        plt.show()
    
        plt.plot(history.history['pearson_correlation'])
        plt.plot(history.history['val_pearson_correlation'])
        plt.title('pearson_correlation')
        plt.ylabel('pearson_correlation')
        plt.xlabel('epoch')
        plt.legend(['Training', 'Validation'], loc='upper left')
        plt.savefig('report/pearson_correlation_'+str(size)+'_B'+str(nBatch)+'_L'+str(learn)+'_D'+str(drop)+'.png')
        plt.show()
        del(model)  # in order to confirm that model weights shall not be utilized in next model
        print("---------")
print("Reached end")
