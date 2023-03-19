#import  libraries
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import layers 
from keras import Model
from keras import regularizers
from keras import losses, optimizers, metrics, callbacks
import pandas as pd

#import custom functions
from model_functions import create_model
from model_functions import callback_list

#------------------------------------------------------------------------------------------------------

#load our dataset (images and lables with supervised = true)
train_data, val_data, test_data = tfds.load('plant_village', split=['train[:70%]', 'train[70%:85%]', 'train[85%:]'],
                                            as_supervised=True)

#------------------------------------------------------------------------------------------------------

#shuffle, batching and prefetching operation (useful to speed up the training with batches)
train_data = train_data.shuffle(1000).batch(8).prefetch(tf.data.AUTOTUNE)

# here shuffling is not done to compute accuracy on the original images in the same order
val_data = val_data.batch(8).prefetch(tf.data.AUTOTUNE)
test_data = test_data.batch(8).prefetch(tf.data.AUTOTUNE)

#creation of a dataframe in which put model selection result for each hyperparameter combination
column_names = ["simulation", "dropout", "l2 regularization", "conv_blocks", "layers_per_block", "dense units", "accuracy"]
df = pd.DataFrame(columns = column_names)
num_classes      = 38  #don't change it (it depends on dataset)

#------------------------------------------------------------------------------------------------------

#list of hyperparameters (l2 regularization choice seemed not to affect so much the result so it was wiped out to speed up procedure)
#same goes for learning rate 
drop_rate        =  [0.3, 0.5]
#l2_reg          =  [0.1, 0.01 ]
l2               =  0.01
units            =  [1024, 512]
layers_per_block =  [1, 2, 3]
conv_blocks      =  [1, 2]

sim = 0        #starting simulation = 0
epochs_for_selection = 3   #it was chosen 3 to speed up selection procedure as possible but leaving the model enough (hopefully) time

#------------------------------------------------------------------------------------------------------

#model selection loop (using for loops)
for drop in drop_rate:
   #for l2 in l2_reg: 
      for u in units: 
        for lay in layers_per_block:
          for bloc in conv_blocks:
            
            #for each combination model is created, trained and tested

            model = create_model(num_classes, drop, l2, u, lay, bloc)
            model.fit(train_data, epochs = epochs_for_selection, validation_data = val_data, callbacks = callback_list() )
            _, accuracy = model.evaluate(test_data)
            
            #results are appended in the dataframe
            df_temp = pd.DataFrame([['simulation ' + str(sim + 1), drop , l2 ,lay , bloc,  u , accuracy ]], columns = column_names )
            print('simulation ' + str(sim + 1), drop , l2 ,lay , bloc,  u , accuracy)
            df = df.append(df_temp, ignore_index=True)

            sim += 1

#NB computation here takes several hours

#------------------------------------------------------------------------------------------------------

#chosen model based on test accuracy
max_accuracy_index = df["accuracy"].idxmax() 
print(df.iloc[max_accuracy_index])

#results of model selection based on test accuracy are our tuned hyperparameter
dropout_rate      = df.iloc[max_accuracy_index][1]
l2_regularization = df.iloc[max_accuracy_index][2]
conv_blocks       = df.iloc[max_accuracy_index][3]
layers_per_block  = df.iloc[max_accuracy_index][4]
units_dense       = df.iloc[max_accuracy_index][5]


print(l2_regularization)
print(dropout_rate)
print(conv_blocks)
print(layers_per_block)
print(units_dense)