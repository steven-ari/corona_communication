import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# show if using CPU or GPU https://www.tensorflow.org/guide/gpu
tf.debugging.set_log_device_placement(True)
# allowing more memory for tensorflow, tensorflow auto use GPU if detected
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# import emnist
# define model and the preprocessing for it
num_classes = 62
img_size = (28, 28)
batch_size = 200
model = Sequential([
    keras.Input(shape=(img_size[0], img_size[1], 1)),
    layers.BatchNormalization(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.BatchNormalization(),
    layers.Conv2D(64, 3, padding='same'),
    layers.LeakyReLU(),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(rate=0.2),
    layers.Dense(num_classes, activation='sigmoid'),
])


# dataset
def prepare(input_ds):
    # img = tf.image.per_image_standardization(img)  # mean=0; var=1
    print(input_ds)
    image = input_ds['image']
    label = input_ds['label']
    print(image)
    print(label)
    rescale = layers.experimental.preprocessing.Rescaling(scale=1./255, input_shape=(1, img_size[0], img_size[1]))
    img = rescale(image)
    img = tfa.image.rotate(img, -90)
    img = tf.image.flip_left_right(img)
    return img, label


AUTOTUNE = tf.data.AUTOTUNE   # automatically set num of parallel worker
train_ds, test_ds = tfds.load('emnist/byclass', split=['train', 'test'])
train_ds = train_ds.map(prepare, num_parallel_calls=AUTOTUNE).shuffle(1000).batch(batch_size)
test_ds = test_ds.map(prepare, num_parallel_calls=AUTOTUNE).shuffle(1000).batch(batch_size)

# training
epochs = 10
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
model.summary()
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=epochs
)

# visualize training result
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


model.save_weights('my_checkpoint')

a = 1