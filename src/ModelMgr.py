import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
import Config
import sys
import math


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

class PrintDot(keras.callbacks.Callback):

  DataSize = 0
  NbrBatchesPerEpoch =  0
  EpochCounter = 0
  TotalBatches = 0
  BatchCounter = 0

  def __init__(self, DataSize):
    self.DataSize = DataSize * (1 - Config.VALIDATION_SPLIT)
    self.NbrBatchesPerEpoch = math.ceil(self.DataSize / Config.BATCH_SIZE)
    self.TotalBatches = self.NbrBatchesPerEpoch * Config.EPOCHS

  def on_epoch_begin(self, epoch, logs):
    self.EpochCounter += 1

  def on_batch_begin(self, batch, logs=None):
    self.BatchCounter += 1
    sys.stdout.flush()
    print(str(int(self.BatchCounter/(self.TotalBatches) * 100)) + " %")



def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()