import tensorflow as tf


class RandomFlip(tf.keras.layers.Layer):
    def __init__(self, seed=42, **kwargs):
        super(RandomFlip, self).__init__(**kwargs)
        self.augment_image = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=seed)
        self.augment_mask = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=seed)

    def call(self, inputs, training=None):
        images, masks = inputs

        if training:
            images = self.augment_image(images)
            masks = self.augment_mask(masks)

        return images, masks


class RandomRotation(tf.keras.layers.Layer):
    def __init__(self, seed=42, factor=(-0.2, 0.2), **kwargs):
        super(RandomRotation, self).__init__(**kwargs)
        self.augment_image = tf.keras.layers.experimental.preprocessing.RandomRotation(factor, seed=seed,
                                                                                       interpolation='nearest')
        self.augment_mask = tf.keras.layers.experimental.preprocessing.RandomRotation(factor, seed=seed,
                                                                                      interpolation='nearest')

    def call(self, inputs, training=None):
        images, masks = inputs

        if training:
            images = self.augment_image(images)
            masks = self.augment_mask(masks)

        return images, masks


class RandomLighting(tf.keras.layers.Layer):
    def __init__(self, brightness_range=0.2, contrast_range=(0.5, 1.2), **kwargs):
        super(RandomLighting, self).__init__(**kwargs)
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def call(self, inputs, training=None):
        images, masks = inputs

        images = tf.image.random_brightness(images, self.brightness_range)
        images = tf.image.random_contrast(images, lower=self.contrast_range[0], upper=self.contrast_range[1])

        return images, masks


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_filters, base_model_name, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.num_filters = num_filters

        self.base_model_name = base_model_name

        self.conv1 = tf.keras.layers.Conv2D(num_filters, 3, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(num_filters, 3, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.keras.layers.Activation("relu")(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.keras.layers.Activation("relu")(x)

        return x


class UpConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_filters, **kwargs):
        super(UpConvBlock, self).__init__(**kwargs)
        self.num_filters = num_filters

        self.conv_transpose1 = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')
        self.conv_block1 = ConvBlock(num_filters)

    def call(self, inputs, skip_features, training=False):
        x = self.conv_transpose1(inputs)
        x = tf.keras.layers.Concatenate()([x, skip_features])
        x = self.conv_block1(x, training=training)

        return x