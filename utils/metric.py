#!/usr/bin/env python

# --- imports -----------------------------------------------------------------
import tensorflow as tf
import numpy as np



def margin_tf(y_pred,y_true, margin=0.4, downweight=0.5):
    """Penalizes deviations from margin for each logit.

    Each wrong logit costs its distance to margin. For negative logits margin is
    0.1 and for positives it is 0.9. First subtract 0.5 from all logits. Now
    margin is 0.4 from each side.

    Args:
      y_true: tensor, one hot encoding of ground truth.
      y_pred: tensor, model predictions in range [0, 1]
      margin: scalar, the margin after subtracting 0.5 from raw_logits.
      downweight: scalar, the factor for negative cost.

    Returns:
      A tensor with cost for each data point of shape [batch_size].
    """
    logits = y_pred - 0.5
    positive_cost = y_true * tf.cast(tf.less(logits, margin),
                                   tf.float32) * tf.pow(logits - margin, 2)
    negative_cost = (1 - y_true) * tf.cast(
      tf.greater(logits, -margin), tf.float32) * tf.pow(logits + margin, 2)
    return tf.reduce_mean(0.5 * positive_cost + downweight * 0.5 * negative_cost)


def IoU_tf(y_pred, y_true):
    """Returns a (approx) IOU score
    intersection = y_pred.flatten() * y_true.flatten()
    Then, IOU =  intersection / (y_pred.sum() + y_true.sum() - intersection)
    Args:
    :param y_pred: predicted labels (4-D array): (N, H, W, 1)
    :param y_true: groundtruth labels (4-D array): (N, H, W, 1)
    :return
    float: IOU score
    """
    threshold = 0.5
    axis = (0, 1, 2, 3)
    smooth = 1e-5
    pre = tf.cast(y_pred > threshold, dtype=tf.float32)
    truth = tf.cast(y_true > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis)  # AND
    union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=axis)  # OR
    batch_iou = (inse + smooth) / (union + smooth)
    iou = tf.reduce_mean(batch_iou)
    return iou


def dice_jaccard_tf(y_pred, y_true):
    """Returns a (approx) dice score
    intesection = y_pred.flatten() * y_true.flatten()
    Then, dice = 2 * intersection / (y_pred.sum() + y_true.sum())
    :param y_pred: predicted labels (4-D array): (N, H, W, 1)
    :param y_true: groundtruth labels (4-D array): (N, H, W, 1)
    :return
        float: dice score
    """
    smooth = 1e-5
    inse = tf.reduce_sum(y_true * y_pred, axis=(0, 1, 2, 3))
    l = tf.reduce_sum(y_true * y_true, axis=(0, 1, 2, 3))
    r = tf.reduce_sum(y_pred * y_pred, axis=(0, 1, 2, 3))
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return dice


def dice_sorensen_tf(y_pred, y_true):
    smooth = 1e-5
    inse = tf.reduce_sum(y_true * y_pred, axis=(0, 1, 2, 3))
    l = tf.reduce_sum(y_true, axis=(0, 1, 2, 3))
    r = tf.reduce_sum(y_pred, axis=(0, 1, 2, 3))
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return dice


def softmax_tf(y_pred, y_true, epsilon=1e-10):
    """
    Computes cross entropy with included softmax - DO NOT provide outputs from softmax layers to this function!

    For brevity, let `x = output`, `z = target`.  The binary cross entropy loss is
    loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    :param y_pred: predicted labels
    :param y_true: groundtruth labels
    :return
        cross entropy with included softmax
    """
    print(y_true)
    print(y_pred)

    # classification
    if len(y_true.shape) <= 2:
        # labels onehot
        if int(y_true.get_shape()[1]) > 1:
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred))
        # labels encoded as integers
        else:
            return tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=y_true, logits=y_pred))
    # segmentation
    else:
        y_true_flat = tf.reshape(y_true, [-1, int(y_true.shape[3])])
        y_pred_flat = tf.reshape(y_pred, [-1, int(y_true.shape[3])])
        return tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y_true_flat, logits=y_pred_flat))


def sigmoid_tf(y_pred, y_true):
    """
    Computes Sigmoid cross entropy
    :param y_pred:
    :param y_true:
    :return:
    """
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))


def hinge_tf(y_pred, y_true):
    """
    Computes Hinge Loss
    :param y_pred: predicted labels
    :param y_true: groundtruth labels
    :return:
        Hinge Loss
    """
    return tf.losses.hinge_loss(labels=y_true, logits=y_pred)


def mse_loss(y_pred, y_true):
    """
    Computes Sum-of-Squares loss
    :param y_pred: predicted labels
    :param y_true: groundtruth labels
    :return:
        Sum-of-Squares loss
    """
    # y_p_s = tf.nn.sigmoid(y_pred)
    # return tf.losses.mean_squared_error(labels=y_true, predictions=y_p_s) + tf.losses.get_regularization_loss()
    y_p_s = tf.nn.sigmoid(y_pred)
    return tf.losses.mean_squared_error(labels=y_true, predictions=y_p_s)


def mse_accuracy(y_pred, y_true):
    return 1 - mse_loss(y_pred, y_true)


def mse_loss_tf(y_pred, y_true):
    """
    Computes Sum-of-Squares loss
    :param y_pred: predicted labels
    :param y_true: groundtruth labels
    :return:
        Sum-of-Squares loss
    """
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    aa_list = ((y_true_flat - y_pred_flat) * (y_true_flat - y_pred_flat))/2
    aa_reshaped = tf.reshape(aa_list, tf.shape(y_true))
    lst = tf.sqrt(tf.cast(tf.reduce_sum(aa_reshaped, axis=1), tf.float16))
    return tf.reduce_mean(tf.cast(lst, dtype=tf.float32))


def percentage_tf(y_pred, y_true):
    """
    Computes percentage of correct predictions
    :param y_pred:
    :param y_true:
    :return:
    """
    print(y_pred)
    print(y_true)
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1)), tf.float32))

