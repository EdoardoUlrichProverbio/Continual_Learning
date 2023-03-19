#import  libraries
import tensorflow as tf
import tensorflow_datasets as tfds
from keras import layers 
from keras import Model
from keras import regularizers
from keras import losses, optimizers, metrics, callbacks

#-------------------------------------------------------------------------

# add the preprocessing phase as a layer in our model
preprocessing_and_augmentation = tf.keras.Sequential([
  #image pixel value are rescaled in (0,1) interval                                                   
  layers.Rescaling(1./255),
  # i also add some data augmentation (they are active only in training mode)
  layers.RandomZoom(height_factor=(0.1, 0.4), width_factor=(0.1, 0.4) ),
  layers.RandomContrast(factor = [0.75, 1.25]),
  tf.keras.layers.RandomFlip()
])

#-------------------------------------------------------------------------

#fonction to build blocks of convolutional layers for our model
# each block is a sequential which contain n conv (layer + batch) with max pooling (2D) at the end [VGG recipe]

#this function buld a block with many layer as the argument
def block_layers(n_layers, filters ):

  block = tf.keras.Sequential()
  for i in range(n_layers):
    #conv layers
    block.add(layers.Conv2D(filters, 3, padding = "same", activation='relu' )) 
    #batch normalization
    block.add(layers.BatchNormalization())

  block.add(layers.MaxPooling2D())

  return block

#-------------------------------------------------------------------------

#this function buld as many blocked as the argument
def num_blocks( number_of_blocks, layers):

  blocks_sequence = tf.keras.Sequential()
  #base filters number
  filters0 = 16

  for i in range(number_of_blocks):
    #let's double the number of filters for each block built
    k = filters0 * (2**i)
    blocks_sequence.add(block_layers(layers ,  k ))

  return blocks_sequence

#final result is a sequential layer ready to be added 

#-------------------------------------------------------------------------

#function to establish leaning rate decay
def progressive_learning_rate(epoch, l_rate):
  if epoch < 10: return l_rate
  else: return l_rate * tf.math.exp(-0.1)

#-------------------------------------------------------------------------

def callback_list():
#A callback is added telling model weather stop or keep going 
  callbacks = [ callbacks.EarlyStopping(
                # since accuracy is more meaningful than loss, we focus on validation 
                monitor='val_sparse_categorical_accuracy',
                #minimum variation required
                min_delta=0.01,
                #here we set the number of consecutive epochs without an increase in accuracy validation
                patience = 3,
                #take weights with which I have the best accuracy
                restore_best_weights=True,
                verbose=1
                ),
                #not proceed further if loss is Na<<<<n
                callbacks.TerminateOnNaN(),
                #implement learning rate decay (for many epochs)
                callbacks.LearningRateScheduler(schedule = progressive_learning_rate, verbose=0)
                ]
  return callbacks

#-------------------------------------------------------------------------

def create_model(num_classes, drop_rate, l2_reg, units, layers_per_block, conv_blocks):
#these are the dense units of final layers
  a = units
  b = int(units/4)

  #final model sequential based with blocks and processing inside
  model= tf.keras.Sequential([    
    layers.Input(shape=(None, None, 3)),                  
    preprocessing_and_augmentation,
    num_blocks(conv_blocks,  layers_per_block ),
    layers.GlobalAveragePooling2D(),
    layers.Dense(a, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
    layers.Dropout(drop_rate),
    layers.Dense(b, activation='relu',  kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
    layers.Dropout(drop_rate),
    layers.Dense(num_classes)
  ])

  model.compile( 
      loss= losses.SparseCategoricalCrossentropy( from_logits = True ),
      # optimizer is adam algorithm, learning rate was modified based on previous test
      optimizer = optimizers.Adam(learning_rate=0.0001),
      metrics   = metrics.SparseCategoricalAccuracy()
  )

  return model