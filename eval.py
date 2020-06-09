from models.point_cls import PointNet
import tensorflow as tf
from data_util import load_data
from models.point_cls import PointNet
import tensorflow as tf
from data_util import load_data
from config import *
from tensorflow.keras import losses, optimizers
from utils.timer import Timer
import matplotlib.pyplot as plt
from provider import categories
import numpy as np
from utils.pc_util import draw_point_cloud, point_cloud_three_views
import os
from datetime import datetime
from sklearn.metrics import ConfusionMatrixDisplay


def prepare_data(x, y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.one_hot(y, NUM_CLASSES, dtype=tf.float32)
    y = tf.squeeze(y, axis=[0])
    return x, y


model = PointNet(num_points=NUM_POINTS, num_classes=NUM_CLASSES)
logdir = "./log/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir)

if __name__ == "__main__":
    data_train, label_train, data_test, label_test = load_data()
    len_test = len(data_test)
    print(len_test)
    correct = 0
    model.load_weights(os.path.join('./trained_models', 'point_net', 'checkpoint'))
    arr = np.random.randint(0, len_test, size=5)
    y_pred = []
    y_true = []
    for step, number in enumerate(range(len_test)):
        x = tf.cast(data_test[number], dtype=tf.float32)
        x = tf.reshape(x, [-1, NUM_POINTS, 3])
        logits = model.predict(x)
        y = tf.argmax(logits, axis=1)
        y = y.numpy()
        # print("Prediction : {}".format(y))
        # print("Label : {}".format(label_test[number]))
        y_pred.append(y[0])
        y_true.append(label_test[number][0])

        if y == label_test[number]:
            correct += 1
            # print('Correct')
        name_prediction = categories[int(y[0])]
        name_label = categories[int(label_test[number][0])]
        with file_writer.as_default():
            image = point_cloud_three_views(data_test[number])
            image = np.reshape(image, (-1, 500, 1500, 1))
            tf.summary.image("Prediction {} , Label {}".format(name_prediction, name_label),
                             image,
                             max_outputs=1,
                             step=step)
    print("Accuracy : {} % " .format(correct / len_test * 100))
