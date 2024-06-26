# -*- coding: utf-8 -*-
"""Identifier and Recipe Gen

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dX-moleg8adXKNWJIL81qzivhSq1fxsa
"""

import os
import numpy as np
import matplotlib.image as mpimg
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, LeakyReLU
from keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['KAGGLE_CONFIG_DIR'] = "~/.kaggle"

# Downloading
!kaggle datasets download -d kritikseth/fruit-and-vegetable-image-recognition -p ./data

# Unzipping
!unzip ./data/fruit-and-vegetable-image-recognition.zip -d ./data

train_dir = "/content/data/train"
test_dir = "/content/data/test"

# Categories
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

# Creating generator by categories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)

# CNN model
model = Sequential()

# Layer 1
model.add(Conv2D(input_shape=(100, 100, 3), kernel_size=(3, 3), filters=32, padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(kernel_size=(3, 3), filters=64, padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2
model.add(Conv2D(kernel_size=(3, 3), filters=128, padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layer for transitioning
model.add(Flatten())

# Dense Layers
model.add(Dense(256))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))

# Final
model.add(Dense(len(train_generator.class_indices), activation='softmax'))

# Compiling
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
tensorboard = TensorBoard(log_dir="logs/{}".format("Vegetable_recognition"))

# Fitting model to the NN
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=1,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[tensorboard]
)

# Evaluate the model on test data using generator
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print("Test Accuracy:", test_accuracy)

# Save the model
model.save('identify.keras')

model.save('identifylegacy.h5')

model.save('identify.json')

"""# **Extra Data Manipulation Tools**"""

# Shuffling data
np.random.shuffle(train_data)
np.random.shuffle(test_data)

# Separating images and labels from data
train_x, train_y = zip(*train_data)
test_x, test_y = zip(*test_data)

# Converting lists to numpy arrays and normalize image data
train_x = np.array(train_x, dtype=np.float32) / 255.0
train_y = np.array(train_y, dtype=np.float32)
test_x = np.array(test_x, dtype=np.float32) / 255.0
test_y = np.array(test_y, dtype=np.float32)

# Defining CNN model
# Layer 1

model = Sequential()

model.add(Conv2D(input_shape=(100, 100, 3), kernel_size=(3, 3), filters=32, padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(kernel_size=(3, 3), filters=64, padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))

# Layer 2
model.add(Conv2D(kernel_size=(3, 3), filters=64, padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Additional Convolutional Layer
model.add(Conv2D(kernel_size=(3, 3), filters=128, padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening layer to transition from convolutional to fully connected layers
model.add(Flatten())

# Dense Layers because why not
model.add(Dense(256))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))

# Final output
model.add(Dense(len(train_categories), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Set up TensorBoard callback
tensorboard = TensorBoard(log_dir="logs/{}".format("Vegetable_recognition"))

# Train the model
model.fit(x=train_x, y=train_y, batch_size=32, epochs=10, callbacks=[tensorboard], validation_split=0.2)

# Save the model
model.save("model.h5")

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(x=test_x, y=test_y, batch_size=32)
print("Test Accuracy:", test_accuracy)