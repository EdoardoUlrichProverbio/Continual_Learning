#import  libraries
import tensorflow as tf
import numpy as np
import pandas as pd

#------------------------------------------------------------------------------------------------------

#since the code deals with one task at a time all lables are rescaled
#obviously floor divisor must be changed depending on the amount of task in the n case
def transform_labels(image, label):
  return image, tf.math.floor(label / 5)

#------------------------------------------------------------------------------------------------------

#function to split data according to labels
def split_data_for_tasks(data, n_tasks):

  list_of_tasks_data = []
#here is used floor operation to build groups of classes for each class
  for i in range(n_tasks):
    list_of_tasks_data.append(data.filter(lambda img, label: label % n_tasks  == i))

  return list_of_tasks_data

#------------------------------------------------------------------------------------------------------

#function to implement shuffling, batching and prefetching operation for each tasks
def data_start_processing (train_bool, data, n_tasks, batch_size, shuffle_size):
#shuffle only train dataset
  if train_bool == True: 
    for i in range(n_tasks):
      data[i] = data[i].map(transform_labels).shuffle(shuffle_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

  else:
    for i in range(n_tasks):
      data[i] = data[i].map(transform_labels).batch(batch_size).prefetch(tf.data.AUTOTUNE)

  return data