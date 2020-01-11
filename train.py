# USAGE
# python train.py --lr-find 1
# python train.py

# set the matplotlib backend so figures can be saved in the background
import os
import random

import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.learningratefinder import LearningRateFinder
from pyimagesearch.firedetectionnet import FireDetectionNet
from pyimagesearch.backend import NerualNetworkModel
from pyimagesearch import config
from utils.DataGenerator import DataGenerator
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import sys
import keras_applications

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--lr-find", type=int, default=0, help="whether or not to find optimal learning rate")
args = vars(ap.parse_args())

# load the smog and non-smog images
print("[INFO] loading data...")
dataset_smog = []
dataset_non_smog = []
for filename in os.listdir(config.SMOG_PATH):
    dataset_smog.append(os.path.sep.join([config.SMOG_PATH, filename]))

for filename in os.listdir(config.NON_SMOG_PATH):
    dataset_non_smog.append(os.path.sep.join([config.NON_SMOG_PATH, filename]))

dataset = dataset_smog + dataset_non_smog
random.shuffle(dataset)
trainDataset = dataset[:int(len(dataset) * 0.8)]
validDataset = dataset[int(len(dataset) * 0.8):]
trainDG = DataGenerator(dataset=trainDataset, batch_size=config.BATCH_SIZE, shuffle=True)
validDG = DataGenerator(dataset=validDataset, batch_size=config.BATCH_SIZE, shuffle=True)

# # initialize the training data augmentation object
# aug = ImageDataGenerator(
#     rotation_range=30,
#     zoom_range=0.15,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.15,
#     horizontal_flip=True,
#     fill_mode="nearest")

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=config.INIT_LR)
# model = FireDetectionNet.build(width=128, height=128, depth=3, classes=2)
# model = ResNet50(include_top=True,
#                  weights=None,
#                  input_shape=(128, 128, 3),
#                  classes=2)

model = NerualNetworkModel().ResNet50(input_size=(128, 128, 3))
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# check to see if we are attempting to find an optimal learning rate
# before training for the full number of epochs

# if args["lr_find"] > 0:
#     # initialize the learning rate finder and then train with learning
#     # rates ranging from 1e-10 to 1e+1
#     print("[INFO] finding learning rate...")
#     lrf = LearningRateFinder(model)
#     lrf.find(
#         aug.flow(trainX, trainY, batch_size=config.BATCH_SIZE),
#         1e-10, 1e+1,
#         stepsPerEpoch=np.ceil((trainX.shape[0] / float(config.BATCH_SIZE))),
#         epochs=20,
#         batchSize=config.BATCH_SIZE,
#         classWeight=classWeight)
#
#     # plot the loss for the various learning rates and save the
#     # resulting plot to disk
#     lrf.plot_loss()
#     plt.savefig(config.LRFIND_PLOT_PATH)
#
#     # gracefully exit the script so we can adjust our learning rates
#     # in the config and then train the network for our full set of
#     # epochs
#     print("[INFO] learning rate finder complete")
#     print("[INFO] examine plot and adjust learning rates before training")
#     sys.exit(0)

# train the network
print("[INFO] training network...")
callbacks = [EarlyStopping(monitor='val_loss', patience=16),
             CSVLogger(config.TRAIN_LOG_PATH),
             # ModelCheckpoint(filepath=os.path.sep.join([config.MODEL_CHECKPOINT_PATH, "model.{epoch:02d}-{val_loss:.2f}.h5"]),
             #                 save_best_only=True,
             #                 period=5)
             ]


H = model.fit_generator(
    generator=trainDG,
    validation_data=validDG,
    epochs=config.NUM_EPOCHS,
    verbose=1,
    callbacks=callbacks,
    workers=4)

# evaluate the network and show a classification report
print("[INFO] evaluating network...")
# predictions = model.predict(testX, batch_size=config.BATCH_SIZE)
# print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=config.CLASSES))

# serialize the model to disk
print("[INFO] serializing network to '{}'...".format(config.MODEL_PATH))
model.save(config.MODEL_PATH)

# construct a plot that plots and saves the training history
N = np.arange(0, len(H.history["loss"]))
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(config.TRAINING_PLOT_PATH)
