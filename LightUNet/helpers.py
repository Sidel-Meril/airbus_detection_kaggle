import tensorflow as tf

class DiceScore(tf.keras.metrics.Metric):
    def __init__(self, smooth=1.0, **kwargs):
        super(DiceScore, self).__init__(name='dice_score', **kwargs)
        self.smooth = smooth
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred > 0, dtype=tf.float32)

        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

        self.intersection.assign_add(tf.reduce_sum(intersection))
        self.union.assign_add(tf.reduce_sum(union))

    def result(self):
        intersection_value = self.intersection
        union_value = self.union
        dice = tf.reduce_mean((2. * intersection_value + self.smooth) / (union_value + self.smooth))
        return dice


def dice_score(y_true, y_pred):
    smooth = 1.0

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred > 0, dtype=tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

    dice = tf.reduce_mean((2. * intersection + smooth) / (union + smooth))
    return dice


class MixedLoss(tf.keras.losses.Loss):
    def __init__(self, alpha, gamma, smooth=1.0, **kwargs):
        super(MixedLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.binary_focal_loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=gamma)

    def call(self, y_true, y_pred, sample_weight=None):
        dice_loss = tf.math.log(dice_score(y_true, y_pred))
        focal_loss = self.binary_focal_loss(y_true, y_pred)

        loss = self.alpha * focal_loss - dice_loss

        return tf.reduce_mean(loss)
