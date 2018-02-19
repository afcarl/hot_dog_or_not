#!/usr/bin/env python3

import os
import sys


import keras
from keras.applications import mobilenet
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential


def load_model():
    origin_model = mobilenet.MobileNet(weights="imagenet") # Will auto dl the model

    output = origin_model.layers[-5] # Before the 1000 classes classification

    output.add(keras.layers.Flatten())
    output = keras.layers.Dense(1, activation='sigmoid')

    model = keras.models.Model(inputs=origin_model.inputs,
                               outputs=output)

    model.compile(
        optimizers=optimizers.Adam(lr=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def preprocess(img):
    if img.ndim == 3:
        img = img[np.newaxis, :, :, :]
    return mobilenet.preprocess_input(img)


def load_data(data_path, *, batch_size):
    data_gen = ImageDataGenerator(preprocessing_function=preprocess)

    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')

    train_flow = data_gen.flow_from_directory(
        train_path,
        target_size=(224, 244),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    return train_flow


def get_size(data_path, kind='train'):
    path = os.path.join(data_path, kind)

    true = os.path.join(path, 'hot_dog')
    false = os.path.join(path, 'not_hot_dog')

    return len(os.listdir(true)) + len(os.listdir(false))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Give the dataset in input!')
        exit(1)

    data_path = sys.argv[1]

    batch_size = 32

    data = load_data(data_path, batch_size=batch_size)
    data_size = get_size(data_path)
    model = load_model()

    model.fit_generator(
        data,
        steps_per_epoch=data_size // batch_size,
        epochs=10,
        verbose=2
    )



