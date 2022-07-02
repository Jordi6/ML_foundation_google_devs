import tensorflow as tf
import numpy as np 
from tensorflow import keras


# Define and complie the neural network
# simple neural network. it has 1 layer, and that layer has 1 nueron.
# and the input shape to it is just 1 value.
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])


# now we compile our nn. we have to specify 2 functions, a loss and an optimizer.
model.compile(optimizer='sgd', loss='mean_squared_error')

# providing the data. we are taking 6xs and 6ys. 
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)


# traning the neural network
model.fit(xs, ys, epochs=500)

# we can use the model.predict method to have it figure out the Y for a previously unknown X.
print("using the model to predict")
print(model.predict([10.0]))
