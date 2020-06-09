import tensorflow as tf 
import numpy as np 
from tensorflow import keras


class Conv2D(keras.layers.Layer):
    def __init__(self,
                 name,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 activation=True,
                 use_bn=True,
                 bn_momentum=0.99):
        super(Conv2D, self).__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.paddding = padding
        self.relu = keras.layers.Activation('relu')
        self.use_bn = use_bn
        self.activation = activation
        self.bn_momentum = bn_momentum
        self.conv2d = keras.layers.Conv2D(filters=self.filters,
                                          kernel_size=self.kernel_size,
                                          strides=self.strides,
                                          padding=self.paddding,
                                          )
        if self.use_bn:
            self.batch_norm = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)

    def call(self, inputs, training=None):
        if len(inputs.shape.as_list()) != 4:
            inputs = tf.expand_dims(inputs, -1)
        out = self.conv2d(inputs)
        if self.use_bn:
            out = self.batch_norm(out, training=training)
        if self.activation:
            out = self.relu(out)
        return out


class FullyConnected(keras.layers.Layer):

    def __init__(self,
                 units,
                 name,
                 activation=True,
                 use_bn=True,
                 bn_momentum=0.99):
        super(FullyConnected, self).__init__(name=name)
        self.use_bn = use_bn
        self.activation = activation
        self.fc = keras.layers.Dense(units)
        self.relu = keras.layers.Activation("relu")
        self.bn_momentum = bn_momentum
        if use_bn:
            self.batch_norm = tf.keras.layers.BatchNormalization(momentum=bn_momentum)

    def call(self, inputs, training=None):
        out = self.fc(inputs)
        if self.use_bn:
            out = self.batch_norm(out, training=training)
        if self.activation:
            out = self.relu(out)
        return out


# class Dense(tf.keras.layers.Layer):
#     def __init__(self, units, activation=tf.nn.relu, use_bn=False, bn_momentum=0.99, **kwargs):
#         super(Dense, self).__init__(**kwargs)
#         self.units = units
#         self.activation = activation
#         self.use_bn = use_bn
#         self.bn_momentum = bn_momentum
#         self.dense = tf.keras.layers.Dense(units, activation=activation, use_bias=not use_bn)
#         if use_bn:
#             self.batch_norm = tf.keras.layers.BatchNormalization(momentum=bn_momentum)
#
#     def call(self, inputs, training=None):
#         x = self.dense(inputs)
#         if self.use_bn:
#             x = self.batch_norm(x, training=training)
#         if self.activation:
#             x = self.activation(x)
#         return x


# class TransformNet(tf.keras.layers.Layer):
#     def __init__(self, add_regularization=True, use_bn=True, bn_momentum=0.95, **kwargs):
#         super(TransformNet, self).__init__(**kwargs)
#         self.w = self.add_weight(shape=(256, self.K**2), initializer=tf.zeros_initializer, trainable=True, name='w')
#         self.b = self.add_weight(shape=(self.K, self.K), initializer=tf.zeros_initializer, trainable=True, name='b')
#         self.eye = tf.constant(np.eye(self.K), dtype=tf.float32)
#         self.add_regularization = add_regularization
#         self.use_bn = use_bn
#         self.bn_momentum = bn_momentum
#         self.conv_0 = Conv2D(64, (1, 1), activation=tf.nn.relu, use_bn=self.use_bn, bn_momentum=self.bn_momentum)
#         self.conv_1 = Conv2D(128, (1, 1), activation=tf.nn.relu, use_bn=self.use_bn, bn_momentum=self.bn_momentum)
#         self.conv_2 = Conv2D(1024, (1, 1), activation=tf.nn.relu, use_bn=self.use_bn, bn_momentum=self.bn_momentum)
#         self.fc_0 = Dense(512, activation=tf.nn.relu, use_bn=self.use_bn, bn_momentum=self.bn_momentum)
#         self.fc_1 = Dense(256, activation=tf.nn.relu, use_bn=self.use_bn, bn_momentum=self.bn_momentum)
#
#     def build(self, input_shape):
#         self.K = input_shape[-1]
#         # Initialize bias with identity
#         self.b.assign(self.eye*0.95)
#         super(TransformNet, self).build(input_shape)
#
#     def call(self, inputs, training=None):                              # BxNx3
#         # x = tf.expand_dims(inputs, axis=3)
#         x = self.conv_0(inputs, training=training)                      # BxNx1x64
#         x = self.conv_1(x, training=training)                           # BxNx1x128
#         x = self.conv_2(x, training=training)                           # BxNx1x1024
#         x = tf.squeeze(x, axis=2)                                       # BxNx1024
#         x = tf.reduce_max(x, axis=1)                                    # Bx1x1024
#         x = self.fc_0(x, training=training)                             # Bx512
#         x = self.fc_1(x, training=training)                             # Bx256
#
#         # Convert to KxK matrix to matmul with input
#         x = tf.expand_dims(x, axis=1)
#         x = tf.matmul(x, self.w)                                        # BxK^2
#         x = tf.reshape(x, (-1, self.K, self.K))                         # BxKxK
#         # Add bias term (initialized to identity matrix)
#         # x += self.b
#         x = tf.math.add(x, self.b)
#         # Add regularization
#         if self.add_regularization:
#             xT_x = tf.matmul(tf.transpose(x, perm=[0, 2, 1]), x)
#             reg_loss = tf.nn.l2_loss(xT_x-self.eye)
#             self.add_loss(0.0005 * reg_loss)
#
#         if len(inputs.shape) == 4:
#             inputs = tf.squeeze(inputs, axis=2)
#         return tf.matmul(inputs, x)


if __name__ == "__main__":
    # Test Conv2D
    layer = Conv2D(filters=6, kernel_size=[1, 3], activation=True, use_bn=True, name='conv2')
    inputs = tf.Variable(initial_value=tf.keras.initializers.glorot_normal()(shape=(2, 10, 3)), trainable=False,
                         name='inputs', dtype=tf.float32)
    x = layer(inputs, training=True)
    print(x.shape)
