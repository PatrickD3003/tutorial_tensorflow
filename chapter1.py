import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
#layers and neurons
#when using tensorflow, define layers using sequential.
#inside the sequential, specify what each layer looks like.
#then define the layer using keras, this time using Dense
#Dense means a set of densely connected neurons,
#every neuron is connected to every neuron in the next layer 
l0 = Dense(units=1, input_shape=[1])
model = Sequential([l0])

#using stochastic gradient descent optimizer
#a complex mathematical function that, when given the values,
#,the previous guess, and the results of calculating the errors on that
#guess, can then generate another one.
model.compile(optimizer='sgd',loss='mean_squared_error')

#format our number into the data format that the layers expect.
#we use Numpy that tensorflow easy to use
xs = np.array([-1.0,0.0,1.0,2.0,3.0,4.0],dtype=float)
ys = np.array([-3.0,-1.0,1.0,3.0,5.0,7.0], dtype =float)

#the learning process begin with the model.fit command
#read this as "fit the Xs to the Ys,and try it 500 times"
model.fit(xs,ys,epochs=500)
print(model.predict([10.0]))
print("Here is what i learned : {}".format(l0.get_weights()))