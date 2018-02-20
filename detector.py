#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np

import keras
from keras.applications import mobilenet
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.utils.generic_utils import CustomObjectScope


def create_model():
    model = mobilenet.MobileNet(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
        ) # Will auto dl the model

    for layer in model.layers:
        layer.trainable = False

    output = Sequential()
    output.add(Flatten(input_shape=model.output_shape[1:]))
    output.add(Dense(256, activation='relu'))
    output.add(Dropout(0.5))
    output.add(Dense(1, activation='sigmoid'))

    model = Model(inputs=model.input, outputs=output(model.output))

    model.compile(
        optimizer=optimizers.Adam(lr=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def preprocess(img):
    if img.ndim == 3:
        img = img[np.newaxis, :, :, :]
    return mobilenet.preprocess_input(img)


def load_data(data_path, *, batch_size):
    data_gen = ImageDataGenerator(
        preprocessing_function=preprocess,
        horizontal_flip=True,
    )

    train_path = os.path.join(data_path, 'train')
    print('Loading {}...'.format(train_path))

    train_flow = data_gen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    return train_flow


def get_size(data_path, kind='train'):
    path = os.path.join(data_path, kind)

    count = 0
    for folder in os.listdir(path):
        folder = os.path.join(path, folder)
        if os.path.isdir(folder) and not folder.startswith('.'):
            count += len(os.listdir(folder))

    return count


def parse_cli(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', dest='train',
                        help='Training mode')
    parser.add_argument('--test', action='store_true', dest='test',
                        help='Testing mode')
    parser.add_argument('--data', action='store', dest='data',
                        help='Indicates data location', required=True)
    parser.add_argument('--model', action='store', dest='model',
                        help='Saves or loads the model', required=True)

    args = parser.parse_args(argv)
    return args


def fit_model(data_path, batch_size):
    data = load_data(data_path, batch_size=batch_size)
    data_size = get_size(data_path)
    model = create_model()

    model.fit_generator(
        data,
        steps_per_epoch=data_size // batch_size,
        epochs=10,
        verbose=1
    )


    return model


def test_model(model, data_path, batch_size):
    test_path = os.path.join(data_path, 'test')

    data_gen = ImageDataGenerator(preprocessing_function=preprocess)
    test_flow = data_gen.flow_from_directory(
        test_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )

    predictions = []
    real_labels = []
    corrects = []
    for i, (X, y) in zip(range(test_flow.n // batch_size), test_flow):
        pred = model.predict(X).ravel()
        predictions.append(pred)
        real_labels.append(y)

        corrects.extend(list(pred > 0.5) == y)

    predictions = np.concatenate(predictions) # Concat all batches together
    real_labels = np.concatenate(real_labels)

    print('Test accuracy: {}'.format(np.mean(corrects)))

    extremums = np.abs(predictions - real_labels).argsort()
    print('5 Easiest:')
    print_extremums(
        predictions[extremums[:5]],
        np.array(test_flow.filenames, dtype=np.object)[extremums[:5]]
    )

    print('5 hardest:')
    print_extremums(
        predictions[extremums[-5:]],
        np.array(test_flow.filenames, dtype=np.object)[extremums[-5:]]
    )

def test_image(model, path):
    img = load_img(path, target_size=(224, 224))
    img = img_to_array(img)
    img = preprocess(img)

    pred = model.predict([img])[0][0]
    if pred < 0.5:
        pred = 1 - pred
        kind = 'hot dog'
    else:
        kind = 'not hot dog'
    
    print('Predicted {} with proba: {} on {}.'.format(kind, pred, path))



def print_extremums(extremums, img_names):
    for img, pred in zip(img_names, extremums):
        print('Img: {}\tPred: {}'.format(img, pred))


def save_model(model, path):
    if not path.endswith('.h5'):
        path = '{}.h5'.format(path)
    model.save(path)


def load_model(path):
    if not path.endswith('.h5'):
        path = '{}.h5'.format(path)
    
    with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
        model = keras.models.load_model(path)
    return model


def main_logic(args):
    batch_size = 32

    if args.train:
        model = fit_model(args.data, batch_size)
        save_model(model, args.model)
    if args.test:
        model = load_model(args.model)
        test_model(model, args.data, batch_size)



if __name__ == '__main__':
    args = parse_cli(sys.argv[1:])
    main_logic(args)
