# Training kmnist using CNN

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.python.client import device_lib
import argparse
import os
import sys
tfds.disable_progress_bar()

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', dest='epochs',
                    default=10, type=int,
                    help='Number of epochs.')

args = parser.parse_args()

print('Python Version = {}'.format(sys.version))
print('TensorFlow Version = {}'.format(tf.__version__))
print('TF_CONFIG = {}'.format(os.environ.get('TF_CONFIG', 'Not found')))
print('DEVICES', device_lib.list_local_devices())

# Define batch size
BATCH_SIZE = 32

# Load the dataset
datasets, info = tfds.load('kmnist', with_info=True, as_supervised=True)

# Normalize and batch process the dataset
ds_train = datasets['train'].map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y)).batch(BATCH_SIZE)


# Build the Convolutional Neural Network
model = tf.keras.models.Sequential([                               
      tf.keras.layers.Conv2D(16, (3,3), activation=tf.nn.relu, input_shape=(28, 28, 1), padding = "same"),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(16, (3,3), activation=tf.nn.relu, padding = "same"),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation=tf.nn.relu),
      # TODO: Write the last layer.
      # Hint: KMNIST has 10 output classes.
       tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

model.compile(optimizer = tf.keras.optimizers.Adam(),
      loss = tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])



# Train and save the model

MODEL_DIR = os.getenv("AIP_MODEL_DIR")

model.fit(ds_train, epochs=args.epochs)

# TODO: Save your CNN classifier. 
# Hint: Save it to MODEL_DIR.
model.save(MODEL_DIR)
