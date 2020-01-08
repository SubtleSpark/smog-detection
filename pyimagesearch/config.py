# import the necessary packages
import os

OUTPUT_PATH = "output"

# initialize the path to the amog and non-amog dataset directories
SMOG_PATH = os.path.sep.join(["/home/test/huangcj/project/smog/smog_data", "smog"])
NON_SMOG_PATH = os.path.sep.join(["/home/test/huangcj/project/smog/smog_data", "non-smog"])
PREDICT_PATH = r"F:\data_set\smog_predict_data"

# initialize the class labels in the dataset
CLASSES = ["Non-Fire", "Fire"]

# define the size of the training and testing split
TRAIN_SPLIT = 0.80
TEST_SPLIT = 0.20

# define the initial learning rate, batch size, and number of epochs
INIT_LR = 1e-2
BATCH_SIZE = 32
NUM_EPOCHS = 50

# set the path to the serialized model after training
# MODEL_PATH = os.path.sep.join([OUTPUT_PATH, "amog_detection.model"])
MODEL_PATH = os.path.sep.join([OUTPUT_PATH, "smog_detection.model"])

# define the path to the output learning rate finder plot and
# training history plot
LRFIND_PLOT_PATH = os.path.sep.join([OUTPUT_PATH, "lrfind_plot.png"])
TRAINING_PLOT_PATH = os.path.sep.join([OUTPUT_PATH, "training_plot.png"])

# define the path to the output directory that will store our final
# output with labels/annotations along with the number of iamges to
# sample
OUTPUT_IMAGE_PATH = os.path.sep.join([OUTPUT_PATH, "examples"])
SAMPLE_SIZE = 50

# 训练记录csv
TRAIN_LOG_PATH = os.path.sep.join([OUTPUT_PATH, "trainlog.csv"])
# checkpoint文件夹
MODEL_CHECKPOINT_PATH = os.path.sep.join([OUTPUT_PATH, "checkpoint"])














