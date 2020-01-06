# USAGE
# python predict_fire.py

# import the necessary packages
from keras.models import load_model
from keras.utils import plot_model
from pyimagesearch import config
from imutils import paths
import numpy as np
import imutils
import random
import cv2
import os

# load the trained model from disk
print("[INFO] loading model...")
# model = load_model(config.MODEL_PATH)
model = load_model(r'F:\Code\DL\smog-detection\smog_detection_resnet50.model')
plot_model(model=model, show_shapes=True)

# grab the paths to the fire and non-fire images, respectively
print("[INFO] predicting...")
firePaths = list(paths.list_images(config.SMOG_PATH))
nonFirePaths = list(paths.list_images(config.NON_SMOG_PATH))

# combine the two image path lists, randomly shuffle them, and sample
# them
imagePaths = firePaths + nonFirePaths
random.shuffle(imagePaths)
imagePaths = imagePaths[:config.SAMPLE_SIZE]

imagePaths = list(paths.list_images(os.path.sep.join([config.PREDICT_PATH, r'smog'])))

# loop over the sampled image paths
count = 0
for (i, imagePath) in enumerate(imagePaths):
    # load the image and clone it
    image = cv2.imread(imagePath)
    output = image.copy()

    # resize the input image to be a fixed 128x128 pixels, ignoring
    # aspect ratio
    image = cv2.resize(image, (128, 128))
    image = image.astype("float32") / 255.0

    # make predictions on the image
    preds = model.predict(np.expand_dims(image, axis=0))[0]
    j = np.argmax(preds)
    label = config.CLASSES[j]

    # draw the activity on the output frame
    output = imutils.resize(output, width=500)
    if label == "Non-Fire":
        text = "Non-Fire"
        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5)
    else:
        text = "WARNING! Fire!"
        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
        print(imagePath)
        count += 1

    # write the output image to disk
    filename = "{}.jpg".format(i)
    p = os.path.sep.join([config.OUTPUT_IMAGE_PATH, filename])
    cv2.imwrite(filename=p, img=output)

print(count)
