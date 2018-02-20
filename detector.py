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
    """Initializes a MobileNet with a custom top.

    The top layers is made of fc layers fine-tuned for the bi-classification
    tasks while the conv layers are left frozen, keeping the ImageNet weights.

    # Returns:
        model: A MobileNet with custom top.
    """
    model = mobilenet.MobileNet(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
        ) # Will auto dl the model

    for layer in model.layers:
        layer.trainable = False # Freeze conv layers

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
    """Preprocesses the image before feeding it to the NN.

    # Arguments:
        img: A numpy array created from the image.

    # Returns:
        imgL A preprocessed image according to model's specification.
    """
    if img.ndim == 3:
        img = img[np.newaxis, :, :, :] # In case of single image
    return mobilenet.preprocess_input(img)


def load_data(data_path, *, batch_size):
    """Loads the data and create a flow from it.

    # Arguments:
        data_path: Path of the data.
                   Must be of the form 'data_path/[test | train]/[true | false]'
        batch_size: Number of images per batch.

    # Returns:
        train_flow: A flow of images, randomly horizontaly flipped.
    """
    assert isinstance(batch_size, int), \
           "batch_size must be an int, not an {}".format(type(batch_size))
    assert batch_size > 0, \
           "batch_size must be > 0, not {}".format(str(batch_size))

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
    """Returns number of element in the dataset.

    # Arguments:
        data_path: Path of the data.
                   Must be of the form 'data_path/[test | train]/[true | false]'
        kind: Whether 'train' or 'test'.

    # Returns:
        count: Number of elements.
    """
    assert kind in ['train', 'test'], \
           "Either use 'train' or 'test', not {}".format(kind)

    path = os.path.join(data_path, kind)

    count = 0
    for folder in os.listdir(path):
        folder = os.path.join(path, folder)
        if os.path.isdir(folder) and not folder.startswith('.'):
            count += len(os.listdir(folder))

    return count


def parse_cli(argv):
    """Parses the command line.

    # Arguments:
        argv: Arguments given in cli excepts [0] (the program's name).

    # Returns:
        args: The parsed arguments.
    """
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
    """Fits the model to the required data.

    # Arguments:
        data_path: Path of the data.
                   Must be of the form 'data_path/[test | train]/[true | false]'
        batch_size: Number of images per batch.

    # Returns:
        model: The trained model.
    """
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
    """Tests the model on a given folder.

    Computes the overall accuracy and provide the 5 easiest examples and the 5
    hardest examples.

    # Arguments:
        model: The trained & fine-tuned model.
        data_path: Path of the data.
                   Must be of the form 'data_path/[test | train]/[true | false]'
        batch_size: The number of images per batch.
    """
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
    """Tests the model on a single image.

    Prints which label has been predicted and its associated confidence.

    # Arguments:
        model: The trained & fine-tuned model.
        path: The image path. Either absolute, or relative to the model.
    """
    img = load_img(path, target_size=(224, 224))
    img = img_to_array(img)
    img = preprocess(img)

    pred = model.predict(img)[0][0]
    if pred < 0.5:
        pred = 1 - pred
        kind = 'hot dog'
    else:
        kind = 'not hot dog'

    print('Predicted {} with proba: {} on {}.'.format(kind, pred, path))



def print_extremums(extremums, img_names):
    """Helper functions to print the extremums (easiest or hardest)."""
    for img, pred in zip(img_names, extremums):
        print('Img: {}\tPred: {}'.format(img, pred))


def save_model(model, path):
    """Saves the model to the desired path.

    If the choosen path does not end by '.h5', it'll be automatically added.

    # Arguments:
        model: The trained & fine-tuned model.
        path: Path where the model will be saved.
    """
    if not path.endswith('.h5'):
        path = '{}.h5'.format(path)
    model.save(path)


def load_model(path):
    """Loads the model located at the given path.

    # Arguments:
        path: Location of the dumped model.

    # Returns:
        model: The loaded model.
    """
    if not path.endswith('.h5'):
        path = '{}.h5'.format(path)

    with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
        model = keras.models.load_model(path)
    return model


def main_logic(args):
    """Main Logic."""
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
