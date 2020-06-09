from models.transformer_nets import InputTransformNet, FeatureTransformNet
import tensorflow as tf 
from tensorflow import keras
from models.layers import FullyConnected, Conv2D


class PointNet(keras.Model):
    """
    PointNet implement for classification
    """

    def __init__(self, num_points, num_classes):
        super(PointNet, self).__init__()
        self.num_points = num_points

        # input
        self.input_transform = InputTransformNet(num_points=num_points)
        self.dot_input = tf.keras.layers.Dot(axes=(2, 1), name='input_dot')
        # mlp
        # self.conv1 = keras.layers.Conv2D(64, [1, 3], padding='valid', strides=(1, 1), name='conv1')
        # self.conv2 = keras.layers.Conv2D(64, [1, 1], padding='valid', strides=(1, 1), name='conv2')
        self.conv1 = Conv2D(filters=64, kernel_size=[1, 3], padding='valid', strides=(1, 1),
                            use_bn=True, activation=True,
                            name='conv1')
        self.conv2 = Conv2D(filters=64, kernel_size=[1, 1], padding='valid', strides=(1, 1),
                            use_bn=True, activation=True,
                            name='conv2')
        # feature
        self.feature_transform = FeatureTransformNet(num_points=num_points)
        self.end_points = {}
        self.dot_feature = tf.keras.layers.Dot(axes=(2, 1), name='output_dot')
        # mlp
        # self.conv3 = keras.layers.Conv2D(64, [1, 1], padding='valid', strides=(1, 1), name='conv3')
        # self.conv4 = keras.layers.Conv2D(128, [1, 1], padding='valid', strides=(1, 1), name='conv4')
        # self.conv5 = keras.layers.Conv2D(1024, [1, 1], padding='valid', strides=(1, 1), name='conv5')

        # mlp
        self.conv3 = Conv2D(filters=64, kernel_size=[1, 1], padding='valid', strides=(1, 1),
                            use_bn=True, activation=True,
                            name='conv3')
        self.conv4 = Conv2D(filters=128, kernel_size=[1, 1], padding='valid', strides=(1, 1),
                            use_bn=True, activation=True,
                            name='conv4')
        self.conv5 = Conv2D(filters=1024, kernel_size=[1, 1], padding='valid', strides=(1, 1),
                            use_bn=True, activation=True,
                            name='conv5')

        # Symmetric function
        self.max_pooling = keras.layers.MaxPool2D([self.num_points, 1], padding='valid', name='point_net_max_pooling')

        self.flatten = keras.layers.Flatten()
        self.fc1 = FullyConnected(units=512, use_bn=True, activation=True, name='fc1')
        self.drop_out1 = keras.layers.Dropout(0.3)
        self.fc2 = FullyConnected(units=256, use_bn=True, activation=True, name='fc2')
        self.drop_out2 = keras.layers.Dropout(0.3)
        self.fc3 = keras.layers.Dense(num_classes, activation=None)

    def call(self, inputs, training=None):
        transform = self.input_transform(inputs)
        # print(transform.shape)
        out = self.dot_input([inputs, transform])
        out = tf.expand_dims(out, -1)
        # print(out.shape)
        out = self.conv1(out)
        out = self.conv2(out)
        transform = self.feature_transform(out)
        self.end_points['transform'] = transform
        out = tf.squeeze(out, axis=[2])
        out = self.dot_feature([out, transform])
        out = tf.expand_dims(out, axis=[2])
        # mlp
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.max_pooling(out)

        out = self.flatten(out)
        out = self.fc1(out)
        out = self.drop_out1(out)
        out = self.fc2(out)
        out = self.drop_out2(out)
        out = self.fc3(out)

        return out

    def get_end_points(self):
        return self.end_points


if __name__ == '__main__':
    model = PointNet(num_points=2048, num_classes=40)
    model.build(input_shape=(None, 2048, 3))
    model.summary()
    x = tf.zeros([32, 2048, 3])
    y = model(x)
    print(y.shape)
