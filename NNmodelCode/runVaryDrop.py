"""
This Code runs all the dropout variations in a predefined NN of specified
Hidden Layer Size and Batch Size on the Scaled Data.
The variations in dropout are measured in the results.csv (by default)
"""

size = 94
nBatch = 64

import numpy
import keras
import pandas
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np

def printData(size, nBatch, history, fileName="results.csv", learnR=0.1, drop=0, test=[0,0]):
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
    out.write(",")
    out.write(str(test[0]))
    out.write(",")
    out.write(str(test[1]))
    out.flush()
    out.close()

#input data
X = pandas.read_csv("X_scaled.csv", header=None, low_memory=False, skiprows=1)
Y = pandas.read_csv("Y.csv", header=None, low_memory=False, skiprows=1)
print("Width of Input Layer:: ", len(X[0]))
X_train = np.asarray(X[:267])
Y_train = np.asarray(Y[:267])
X_test = np.asarray(X[267:])
Y_test = np.asarray(Y[267:])
del(X)
del(Y)

#HYPERPARAMETERS
numpy.random.seed(7)
nEpoch = 25
drop = 0.1
activName = 'hard_sigmoid'
initializerName = 'normal'
learn = 0.01
loss = 'mean_absolute_error'

# create list of dropout values ranging from 0 to 1 with 0.01 increments
drop1 = []
for i in range(101):
    drop1.append(float(i * 0.01))

out = open("results/results.csv", "a")
out.write("\n")
out.write("Neuron Number Hidden Layer 1,nBatch,drop,learning Rate,Max Training Loss, Min Training Loss, Avg Training Loss, ")
out.write("Max Correlation Training, Min Correlation Training, Avg Correlation Training,")
out.write("Max Validation Loss, Min Validation Loss, Avg Validation Loss,")
out.write("Max Correlation Validation, Min Correlation Validation, Avg Correlation Validation,")
out.write("Testing Loss, Testing Correlation")
out.flush()
out.close()

for drop in drop1:
    print("Hidden Layer 1 Size", size, "nBatch ", nBatch, "learning rate=", learn, "drop ", drop)
    #create model
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

    #training, validating
    history = model.fit(X_train, Y_train, batch_size=nBatch, epochs=nEpoch, verbose=0, validation_split=0.25, shuffle=True, callbacks=[redLRplaateau, csvLogger])

    #testing and sending to stdout
    data = model.evaluate(X_test, Y_test, batch_size=nBatch, verbose=0)

    printData(size=size, nBatch=nBatch,history=history, learnR=learn, drop=drop,fileName="results/results.csv", test=data)

    #graphical representation of history callback
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.savefig('report/loss_'+str(size)+'_B'+str(nBatch)+'_L'+str(learn)+'_D'+str(drop)+'.png')
    plt.show()

    print("\n\n MODEL, PREDICTION ", data)

    plt.plot(history.history['pearson_correlation'])
    plt.plot(history.history['val_pearson_correlation'])
    plt.title('pearson_correlation')
    plt.ylabel('pearson_correlation')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.savefig('report/pearson_correlation_'+str(size)+'_B'+str(nBatch)+'_L'+str(learn)+'_D'+str(drop)+'.png')
    plt.show()
    del(model)
    print("---------")
print("Reached end")
