import tensorflow as tf
from tensorflow import keras
from models.layers import FullyConnected, Conv2D
import numpy as np


# BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class InputTransformNet(keras.layers.Layer):
    def __init__(self, num_points, K=3):
        super(InputTransformNet, self).__init__(name='input_transform_net')
        self.K = K
        self.num_points = num_points

        self.conv1 = Conv2D(filters=64, kernel_size=[1, 3], padding='valid', strides=(1, 1),
                            use_bn=True, activation=True,
                            name='input_transform_conv1')
        self.conv2 = Conv2D(filters=128, kernel_size=[1, 1], padding='valid', strides=(1, 1),
                            use_bn=True, activation=True,
                            name='input_transform_conv2')
        self.conv3 = Conv2D(filters=1024, kernel_size=[1, 1], padding='valid', strides=(1, 1),
                            use_bn=True, activation=True,
                            name='input_transform_conv3')
        self.maxpooling = keras.layers.MaxPool2D([self.num_points, 1], padding='valid',
                                                 name='input_transform_max_pooling')
        self.flatten = keras.layers.Flatten()
        # fully connected
        self.fc1 = FullyConnected(units=512,
                                  use_bn=True, activation=True,
                                  name='input_transform_fc1')
        self.fc2 = FullyConnected(units=256,
                                  use_bn=True, activation=True,
                                  name='input_transform_fc2')

        self.w = tf.Variable(initial_value=tf.zeros([256, 3 * self.K]), dtype=tf.float32)
        self.b = tf.Variable(initial_value=tf.zeros([3 * self.K]), dtype=tf.float32)
        self.b.assign_add(tf.constant([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=tf.float32))

    def call(self, inputs, training=None):
        inputs = tf.expand_dims(inputs, -1)
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.maxpooling(out)

        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc2(out)

        out = tf.matmul(out, self.w) + self.b
        transform = tf.reshape(out, [-1, 3, self.K])

        return transform


class FeatureTransformNet(keras.layers.Layer):

    def __init__(self, num_points, K=64, add_regularization=True, reg_weight=0.005):
        super(FeatureTransformNet, self).__init__(name='feature_transform_net')
        self.K = K
        self.num_points = num_points
        self.add_regularization = add_regularization
        self.reg_weight = reg_weight
        self.conv1 = Conv2D(filters=64, kernel_size=[1, 1], padding='valid', strides=(1, 1),
                            use_bn=True, activation=True,
                            name='feature_transform_conv1')
        self.conv2 = Conv2D(filters=128, kernel_size=[1, 1], padding='valid', strides=(1, 1),
                            use_bn=True, activation=True,
                            name='feature_transform_conv2')
        self.conv3 = Conv2D(filters=1024, kernel_size=[1, 1], padding='valid', strides=(1, 1),
                            use_bn=True, activation=True,
                            name='feature_transform_conv3')
        self.maxpooling = keras.layers.MaxPool2D([self.num_points, 1], padding='valid',
                                                 name='feature_transform_max_pooling')
        self.flatten = keras.layers.Flatten()
        # fully connected
        self.fc1 = FullyConnected(units=512,
                                  use_bn=True, activation=True,
                                  name='feature_transform_fc1')
        self.fc2 = FullyConnected(units=256,
                                  use_bn=True, activation=True,
                                  name='feature_transform_fc2')

        self.w = tf.Variable(initial_value=tf.zeros([256, self.K * self.K]), dtype=tf.float32)
        self.b = tf.Variable(initial_value=tf.zeros([self.K * self.K]), dtype=tf.float32)
        self.eye = tf.constant(np.eye(self.K), dtype=tf.float32)
        self.b.assign_add(tf.constant(np.eye(K).flatten(), dtype=tf.float32))

    def call(self, inputs, training=None):

        # inputs = tf.expand_dims(inputs, axis=-1)
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.maxpooling(out)

        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc2(out)

        out = tf.matmul(out, self.w) + self.b
        transform = tf.reshape(out, [-1, self.K, self.K])

        if self.add_regularization:
            xT_x = tf.matmul(tf.transpose(transform, perm=[0, 2, 1]), transform)
            reg_loss = tf.nn.l2_loss(xT_x - self.eye)
            self.add_loss(self.reg_weight * reg_loss)

        return transform


if __name__ == '__main__':
    print("Test..")
    layer = FeatureTransformNet(num_points=2048)
    x = tf.zeros([32, 2048, 3])
    y = layer(x)
    print(y.shape)
    x1 = tf.Variable(initial_value=tf.keras.initializers.glorot_normal()(shape=(2, 10, 3)),
                     trainable=False,
                     name='inputs',
                     dtype=tf.float32)
    x2 = tf.Variable(initial_value=tf.keras.initializers.glorot_normal()(shape=(2, 3, 3)),
                     trainable=False,
                     name='inputs',
                     dtype=tf.float32)
    dot = tf.keras.layers.Dot(axes=(2, 1))([x1, x2])
    print(dot.shape)
