import os
from os import path, makedirs

# This next line needs to happen BEFORE tf and keras.
os.environ['CUDA_VISIBLE_DEVICES']='0'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image

import argparse
import numpy as np

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.compat.v1.Session(config=config)

################################################################################
# Settings
################################################################################
image_size = (640, 640)
image_size = (160, 160)
batch_size = 8
epochs = 30


################################################################################
# Data Prep
################################################################################
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/images",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical",
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/images",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical",
)

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)


################################################################################
# Keras model
################################################################################
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

model = make_model(input_shape=image_size + (3,), num_classes=112)
keras.utils.plot_model(model, show_shapes=True)


################################################################################
# Training
################################################################################
parser = argparse.ArgumentParser(description='Train model.')
parser.add_argument("-n", "--name", help = "Model name.", 
    required = False, default = "model0")
parser.add_argument("-p", "--process", help = "Training or Prediction.", default = "training")
args = parser.parse_args()

if not path.isdir("model_checkpoints/" + args.name):
	makedirs("model_checkpoints/" + args.name)

callbacks = [
    keras.callbacks.ModelCheckpoint("model_checkpoints/" + args.name + "/save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

if args.process == 'training':
    model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
    )

else:
    model.load_weights('model_checkpoints/model0/save_at_28.h5')

    # # predicting images
    # img = image.load_img('data/images/ZMB/-12.96906221319054_28.63286521445917.jpg', target_size=image_size)
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)

    # images = np.vstack([x])
    # classes = model.predict(images, batch_size=10)

    # predicting multiple images at once
    img_list = []
    for f in os.listdir('data/images/USA'):
        img = image.load_img(f'data/images/USA/{f}', target_size=image_size)
        y = image.img_to_array(img)
        y = np.expand_dims(y, axis=0)

        img_list.append(y)

    # pass the list of multiple images np.vstack()
    images = np.vstack(img_list)
    classes = model.predict(images, batch_size=10)

    subfolders = [ f.path for f in os.scandir('data/images/') if f.is_dir() ]

    for c in classes:
        print(os.path.basename(subfolders[c.argmax(axis=0)]))