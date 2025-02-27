import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import keras
from keras import  layers
print(tf.__version__)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")
print(x_train.shape)
print(y_train.shape)

x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

# Sequential API (Very convenient, not very flexible)
model1 = keras.Sequential(
    [
        keras.Input(shape=(28*28,)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10)
    ]
)

# Get output from layers (useful for debugging)
model1 = keras.Model(inputs=model1.inputs,
                     outputs=[layer.output for layer in model1.layers])

features = model1.predict(x_train)
for feature in features:
    print(feature.shape)
# Functional API (A bit more flexible)
inputs = keras.Input(shape=(28*28,))
x = layers.Dense(512, activation='relu')(inputs)
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model2 = keras.Model(inputs=inputs, outputs=outputs)

print(model2.summary)

model2.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adagrad(learning_rate=0.001),
    metrics=["accuracy"]
)

model2.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
model2.evaluate(x_test, y_test, batch_size=32, verbose=2)