#import  libraries
import tensorflow as tf

#------------------------------------------------------------------------------------------------------

#function to compute the number of batch in a dataset instance
def number_of_batches(task_dataset):
  num_elements = 0
  for element in task_dataset:
    num_elements += 1
  return num_elements

#------------------------------------------------------------------------------------------------------

def Fisher_I_matrix(model, task_dataset, fmatrix ):
  #fmatrix is passed so every time the new fisher matrix can be added to the previous one
  # computing number of batches 
  num = number_of_batches(task_dataset)

  for i, (images, labels) in enumerate(task_dataset.take(num)):
    #  looking at gradients of model params
    with tf.GradientTape() as tape:
      # computing the log likelihood of  model predictions for each image 
      log_likelihood = tf.nn.log_softmax(model(images))
    # Attach gradients of log likelihood 
    log_L_grads = tape.gradient(log_likelihood, model.trainable_variables)
    for i, grad in enumerate(log_L_grads):   # Compute Fisher Information Matrix as mean of gradients squared
      # so we are passing to a vector
      fmatrix[i] += tf.reduce_mean(grad**2, axis=0)/num 
 
  return fmatrix

#------------------------------------------------------------------------------------------------------

#this is the loss regularized with elastic weight consolidation (ewc)
def elastic_weight_consolidation(labels, preds, model, Fisher, thetabis_i, lambda_par):
  # here below the loss computed without concerning about previous training (without correction)
  previous_loss = model.loss(labels, preds)
  correction = 0
  for i, theta_i in enumerate(model.trainable_variables):
    #computing the regularization like in the formula
    i_value = tf.math.reduce_sum(Fisher[i] * (theta_i - thetabis_i[i]) ** 2)
    correction += i_value
    #lambda values represent how much weight is given to regularization parameter
  correction = lambda_par*correction
  
  return previous_loss + correction

#------------------------------------------------------------------------------------------------------

# Define metrics here:
#it's defined as a function in the R_matrix, which is the matrix of accuracy of task i after training on task j
def backward_transfer(R_matrix):
  #T is number of tasks
  T = len(R_matrix[0])
  BT = 0
  
  for i in range (0, T-1):
    BT += R_matrix[T-1][i]- R_matrix[i][i]

  BT = (1/(T-1))*BT
  return BT
