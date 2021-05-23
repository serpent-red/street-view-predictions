import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'

################################################################################
# Data Prep
################################################################################
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

image_size = (640, 640)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/images",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/images",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

print(val_ds)

# ################################################################################
# # Keras model
# ################################################################################
