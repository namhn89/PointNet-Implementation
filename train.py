from models.point_cls import PointNet
import tensorflow as tf
from data_util import load_data
from config import *
from tensorflow.keras import losses, optimizers
from utils.timer import Timer
import numpy as np
import datetime

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def prepare_data(x, y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.one_hot(y, NUM_CLASSES, dtype=tf.float32)
    y = tf.squeeze(y, axis=[0])
    return x, y


def modelnet40_load_data():
    x_train, y_train, x_test, y_test = load_data()
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_dataset = train_dataset.map(prepare_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.map(prepare_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    len_train = len(x_train)
    train_dataset = train_dataset.shuffle(len_train).batch(PARAMS['batch_size'])
    test_dataset = test_dataset.batch(PARAMS['batch_size'])
    return train_dataset, test_dataset


def get_bn_decay():
    BN_INIT_DECAY = 0.5
    BN_DECAY_RATE = 0.5
    BN_DECAY_CLIP = 0.99
    BATCH_SIZE = PARAMS['batch_size']
    BN_DECAY_STEP = PARAMS['bn_decay_step']

    bn_momentum = tf.keras.optimizers.schedules.ExponentialDecay(
        BN_INIT_DECAY,
        PARAMS['max_epoch'] * BATCH_SIZE,
        BN_DECAY_STEP,
        BN_DECAY_RATE,
    )
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1.0 - bn_momentum)
    return bn_decay


model = PointNet(num_points=NUM_POINTS, num_classes=NUM_CLASSES)
learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
    PARAMS['learning_rate'],
    PARAMS['max_epoch'] * PARAMS['batch_size'],
    PARAMS['decay_rate'],
    staircase=True
)
optimizer = optimizers.Adam(learning_rate_fn)

loss_fn = losses.CategoricalCrossentropy(from_logits=True)
loss_eval = losses.CategoricalCrossentropy(from_logits=True)

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.CategoricalAccuracy('train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.CategoricalAccuracy('test_accuracy')

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = './log/gradient_tape/' + current_time + '/train'
test_log_dir = './log/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


@tf.function
def compute_loss(inputs, labels):
    with tf.GradientTape() as tape:
        logits = model(inputs, training=True)
        loss_1 = loss_fn(y_true=labels, y_pred=logits)
        loss_2 = sum(model.losses)
        loss = loss_1 + loss_2

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_accuracy(y_true=labels, y_pred=logits)
    train_loss(loss)
    return loss


@tf.function
def eval_validation(inputs, labels):
    logits = model(inputs, training=False)
    loss_1 = loss_eval(y_true=labels, y_pred=logits)
    loss_2 = sum(model.losses)
    loss = loss_1 + loss_2
    test_loss(loss)
    test_accuracy(y_true=labels, y_pred=logits)
    return loss


@tf.function
def compute_loss_v2(inputs, labels, end_points, reg_weight=0.001):
    with tf.GradientTape() as tape:
        logits = model(inputs, training=True)
        loss_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        classify_loss = tf.reduce_mean(loss_1)
        tf.summary.scalar('classify loss', classify_loss)
        transform = end_points['transform']
        K = transform.get_shape()[1].value
        mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0, 2, 1]))
        mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
        mat_diff_loss = tf.nn.l2_loss(mat_diff)
        tf.summary.scalar('mat loss', mat_diff_loss)
        loss = classify_loss + reg_weight * mat_diff_loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_accuracy(y_true=labels, y_pred=logits)
    return loss


def main():
    time = Timer()
    model.build(input_shape=(None, NUM_POINTS, 3))
    model.summary()
    time.start(job="Start Training...", verbal=True)
    train, test = modelnet40_load_data()
    best_acc = 0.0
    export_path = os.path.join('./trained_models', 'point_net', 'checkpoint')
    for epoch in range(PARAMS['max_epoch']):
        time.start(job='Epoch {} : '.format(epoch), verbal=True)
        for step, (x, y) in enumerate(train):
            compute_loss(inputs=x, labels=y)
            if step % 100 == 0:
                print('Training loss at step {}: {}, accuracy: {}'.format(step,
                                                                          float(train_loss.result()),
                                                                          float(train_accuracy.result())))

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        print('Training loss at epoch {}: {}'.format(epoch, float(train_loss.result())))
        print('Training accuracy at epoch {}: {}'.format(epoch, float(train_accuracy.result())))

        time.stop()
        time.start(job="Validating at epoch {}".format(epoch), verbal=True)
        for step, (inputs, labels) in enumerate(test):
            eval_validation(inputs, labels)

        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

        print('Validation loss at epoch {}: {}'.format(epoch, float(test_loss.result())))
        print('Validation accuracy at epoch {}: {}'.format(epoch, float(test_accuracy.result())))

        time.stop()

        if test_accuracy.result() > best_acc:
            print('Save model ... ')
            best_acc = test_accuracy.result()
            model.save_weights(export_path)

        train_accuracy.reset_states()
        train_loss.reset_states()
        test_accuracy.reset_states()
        test_loss.reset_states()

    time.stop()


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpus))
    tf.config.experimental_run_functions_eagerly(True)
    tf.config.set_soft_device_placement(True)

    # tf.config.experimental.set_virtual_device_configuration(gpus[0],
    #                                                             [tf.config.experimental.VirtualDeviceConfiguration(
    #                                                                 memory_limit=5000)])
    # tf.debugging.set_log_device_placement(True)

    main()
