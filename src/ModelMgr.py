import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def BuildModel(TrainData):
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(TrainData.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model