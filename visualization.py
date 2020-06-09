from utils.pc_util import point_cloud_three_views
from datetime import datetime
import tensorflow as tf
import numpy as np

from data_util import load_data

logdir = "./log/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir)
x_train, y_train, x_test, y_test = load_data()
train_images = []
for i in range(25):
    train_images.append(point_cloud_three_views(x_train[i]))

print(np.asarray(train_images).shape)

# visualize images
with file_writer.as_default():
    # Don't forget to reshape.
    images = np.reshape(train_images[0:25], (-1, 500, 1500, 1))
    tf.summary.image("25 training data examples", images, max_outputs=25, step=0)
