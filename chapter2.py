import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy')>0.95):
            print("\nReached 95% accuracy so cancelling training! ")
            self.model.stop_training = True
    
callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist

(training_images,training_labels),(test_images,test_labels) = mnist.load_data()

#dividing by 255--> every pixel is represented by a number between 0 and 1 instead.
training_images = training_images/255.0
test_images = test_images/255.0

#Flatten is an input layer specification.
#we want them to be treated as a series of numeric values. 
#flatten takes a 2D array and turns it into a 1D array(a line)
##########################################################################################
#the activation function is code that will execute on each neuron in the layer. 
# relu stands for rectified linear unit. It returns a value if it's greater than 0.
#last dense layer have 10 neurons because we have 10 classes.each of these neurons will end up with a probability that the input piels match that class.
#Softmax takes a set of values, and effectively picks the biggest one.
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
loss = 'sparse_categorical_crossentropy',
metrics=['accuracy'])

model.fit(training_images,training_labels,epochs=50,callbacks=[callbacks])
"""
model.evaluate(test_images,test_labels)
classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])
"""


