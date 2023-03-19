#import  libraries
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import Model
import numpy as np
import pandas as pd

#import custom functions
from model_functions import create_model
from model_functions import callback_list
from data_processing_functions import transform_labels
from data_processing_functions import split_data_for_tasks
from data_processing_functions import data_start_processing
from ewc_functions import number_of_batches
from ewc_functions import Fisher_I_matrix
from ewc_functions import elastic_weight_consolidation
from ewc_functions import backward_transfer


#NB pip install tfds-nightly to find "plant_village" dataset

#------------------------------------------------------------------------------------------------------

#this was the actual result from our model selection
#needs to be commented if we want to run model selection again
dropout_rate      = 0.3
l2_regularization = 0.01
conv_blocks       = 2
layers_per_block  = 1
units_dense       = 512

num_classes = 38

#------------------------------------------------------------------------------------------------------

#load data
train_data, val_data, test_data = tfds.load('plant_village', split=['train[:70%]', 'train[70%:85%]', 'train[85%:]'],
                                            as_supervised=True)

#best model now is used for continual learning (select numbero of epochs)
model = create_model(num_classes, dropout_rate, l2_regularization, units_dense, layers_per_block, conv_blocks)
model.fit(train_data, epochs = 20, validation_data = val_data, callbacks = callback_list )
_, accuracy = model.evaluate(test_data)

# Split the dataset
n_tasks = 5 

#both train, validation and test data are splitted according to their original class
train_data_task  = split_data_for_tasks(train_data, n_tasks)
val_data_task    = split_data_for_tasks(val_data, n_tasks)
test_data_task  = split_data_for_tasks(test_data, n_tasks)

#Vector of task tags
tags = ["A","B","C","D", "E"]

#------------------------------------------------------------------------------------------------------


#partial preprocessing of data. New labels are applied before data is shuffled, batched and prefetched.
#NB only train data is shuffled as before

batch_size    = 16
shuffle_size  = 1000

#vector of train data splitted in task
train_tasks  = data_start_processing ( True, train_data_task, n_tasks, batch_size, shuffle_size)
val_tasks    = data_start_processing ( False, val_data_task, n_tasks, batch_size, shuffle_size)
test_tasks   = data_start_processing ( False, test_data_task, n_tasks, batch_size, shuffle_size)

#------------------------------------------------------------------------------------------------------

# creating r_matrix described above in the function 
R_matrix = np.zeros( (n_tasks, n_tasks) )

num_classes = 8   #max number of classes for each task

#best model now is used for continual learning
model = create_model(num_classes, dropout_rate, l2_regularization, units_dense, layers_per_block, conv_blocks)

#train model initially only on task A
model.fit(train_tasks[0], epochs=5, validation_data=val_tasks[0], callbacks = callback_list )
result = model.evaluate(test_tasks[0])

#evaluation on task A after training on Task A is put as the first value in the matrix
R_matrix[0][0] = result[1]


#this is the section of the various hyperparameters of the following training loop
#as said lambda_par is the weight of the regularization
lambda_par = 0.6

#accuracy and loss are the same than above
accuracy = tf.keras.metrics.SparseCategoricalAccuracy("Accuracy")
loss = tf.keras.metrics.SparseCategoricalCrossentropy("Loss")
epochs = 5

#Fisher information Matrix is built on matrix of trainable parameter of the model 
#in its current state (post task A training)
matrix_base = {n: tf.zeros_like(p.value()) for n, p in enumerate(model.trainable_variables)}
Fisher = Fisher_I_matrix(model, train_tasks[0], matrix_base )


#------------------------------------------------------------------------------------------------------


#for loop for every task (starting from the second one)
#this loop is actually running 3 task but can be used for n task also

for i in range (1,len(tags)):
  #printing the actual training Task
  print("Task: ",tags[i])
  #copy of the current trainable variables tensor
  theta = {n: p.value() for n, p in enumerate(model.trainable_variables.copy())}


  # add new fisher matrix for each new task training (starting from the third one)
  #i tried also to sum fisher matrix from previous tasks but it seems working worse
  if (i > 1): 
    matrix_base = {n: tf.zeros_like(p.value()) for n, p in enumerate(model.trainable_variables)}
    Fisher = Fisher_I_matrix(model, train_tasks[i-1], matrix_base )
  
  # Now we set up the training loop for task i with EWC
  for epoch in range(epochs):
    for batch, (images, labels) in enumerate(train_tasks[i]):
    
      with tf.GradientTape() as tape:
          # Compute predictions and EWC loss (loss + regularization)
          output = model(images)
          EWC = elastic_weight_consolidation(labels, output, model, Fisher, theta, lambda_par)
      # Gradients of trainable parameters are computed with respect to EWC loss
      grads = tape.gradient(EWC, model.trainable_variables)
      # Update  gradients
      model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
      # Update loss and accuracy
      accuracy.update_state(labels,  output)
      loss.update_state(labels,  output)
    #let's print each epoch final status
    print("\rEpoch: {}, Loss: {:.3f}, Accuracy: {:.3f}".format(
        epoch+1, loss.result().numpy(), accuracy.result().numpy()), flush=True, end='\n')

  # at the end of a task training model is evaluated also on previous tasks  
  for j in range (0,i+1):
    
    print("\n")
    print( " Performance on task ",tags[j], "after training on", tags[i], ": \n",  )
    result = model.evaluate(test_tasks[j])
    #every accuracy about task i after training j is put into R_matrix
    R_matrix[i][j] = result[1]
    print("\n")


#------------------------------------------------------------------------------------------------------


#matrix of all accuracies (obviously above the diagonal they are all zeros)
#row are task A,B,C,D,E currently just trained
#columns represent which task is evaluating

df = pd.DataFrame(data = R_matrix, index =["Task A", "Task B", "Task C"," Task D" , "Task E"], 
                  columns =["Task A", "Task B", "Task C"," Task D" , "Task E"])
  
print(" Table of Accuracy on Task i (rows) after training of Task j (columns) ")
df

#it seems tha accuracy of task A,B,D is somewhat mainteined and task E is learned as well
#while task C suffers of cathastrophic forgetting
#probably factor lambda needs to be adjusted


#mean accuracy (as seen in the slides)
M_acc = 2*(R_matrix.sum())/(n_tasks*(n_tasks+1))
print(M_acc)
#the  mean accuracy is very similar to the result obtained training all classes together

#------------------------------------------------------------------------------------------------------

#backward transfer of our model (5 task but program is written to be potentially extended to n)
print(backward_transfer(R_matrix))

#there exists negative backward transfer when learning about some task j
#decreases the performance on some preceding task i
#in this case is obtained a not so negative value which denotes a small degree of forgetting (this result may be improved tuning lambda_par hyperparameter)