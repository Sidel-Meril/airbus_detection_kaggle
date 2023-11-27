from LightUNet.layers import *
from helpers import dice_score
import tensorflow as tf
import os

class LightUNet(tf.keras.Model):
    def __init__(self, base_model_name, checkpoints_path='./', name='light_unet'):
        """
        Light UNet model based on ResNet50V2 or VGG16 as the encoder.

        Parameters:
        - base_model_name (str): Name of the base model. Choose from 'resnet50v2' or 'vgg16'.
        - checkpoints_path (str): Path to save and load model checkpoints.

        Raises:
        - ValueError: If an invalid base_model_name is provided.
        """

        super(LightUNet, self).__init__(name=name)
        self.base_model_name = base_model_name
        self.checkpoints_path = checkpoints_path
        self.name = name

    def build(self, input_shape):
        input_image_shape, input_mask_shape = input_shape['image'], input_shape['mask']
        image = tf.keras.layers.Input(shape=input_image_shape, name='image')
        mask = tf.keras.layers.Input(shape=input_mask_shape, name='mask')

        self.data_augmentation = tf.keras.Sequential([
            RandomRotation(),
            RandomFlip(),
            RandomLighting()
        ])

        if self.base_model_name == 'resnet50v2':

            base_model = tf.keras.applications.resnet_v2.ResNet50V2(input_tensor=image,
                                                                    include_top=False,
                                                                    weights='imagenet')
            base_model.trainable = False

            skip_connections_ = [
                base_model.get_layer("conv3_block4_1_relu").output,  # shape=(48, 48, 512)
                base_model.get_layer("conv2_block3_1_relu").output,  # shape=(96, 96, 256)
                base_model.get_layer("conv1_conv").output,  # shape=(192, 192, 64)
                base_model.get_layer("image").output  # shape=(384, 384, 3)
            ]

            bridge_ = base_model.get_layer("conv4_block6_1_relu").output  # 24

        elif self.base_model_name == 'vgg16':

            base_model = tf.keras.applications.vgg16.VGG16(input_tensor=image,
                                                           include_top=False,
                                                           weights='imagenet')

            base_model.trainable = False

            skip_connections_ = [
                base_model.get_layer("block4_conv3").output,  # shape=(48, 48, 512)
                base_model.get_layer("block3_conv3").output,  # shape=(96, 96, 256)
                base_model.get_layer("block2_conv2").output,  # shape=(192, 192, 64)
                base_model.get_layer("image").output  # shape=(384, 384, 3)
            ]

            bridge_ = base_model.get_layer("block5_conv3").output  # 24
        else:
            raise Exception(f'Base model {self.base_model_name} is not defined. \nChoose from ["resnet50v2","vgg16"]')

        self.skip_connections = tf.keras.Model(inputs=base_model.input, outputs=skip_connections_, trainable=True)
        self.bridge = tf.keras.Model(inputs=base_model.input, outputs=bridge_, trainable=True)

        self.up_conv1 = UpConvBlock(384)
        self.up_conv2 = UpConvBlock(192)
        self.up_conv3 = UpConvBlock(96)
        self.up_conv4 = UpConvBlock(48)

        self.output_conv = tf.keras.layers.Conv2D(48, 3, padding='same', activation='relu')
        self.output_layer = tf.keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')

        super().build({'image': (None, *input_shape['image']), 'mask': (None, *input_shape['mask'])})

    def call(self, inputs, training=True):
        image, mask = inputs['image'], inputs['mask']
        if training: image, mask = self.data_augmentation(inputs=(image, mask), training=training)
        skip_connections, bridge = self.skip_connections(image), self.bridge(image)
        x = self.up_conv1(bridge, skip_connections[0], training=training)
        x = self.up_conv2(x, skip_connections[1], training=training)
        x = self.up_conv3(x, skip_connections[2], training=training)
        x = self.up_conv4(x, skip_connections[3], training=training)
        x = self.output_conv(x)

        outputs = self.output_layer(x)

        return outputs

    def predict(self, image):
        skip_connections, bridge = self.skip_connections(image), self.bridge(image)
        x = self.up_conv1(bridge, skip_connections[0], training=False)
        x = self.up_conv2(x, skip_connections[1], training=False)
        x = self.up_conv3(x, skip_connections[2], training=False)
        x = self.up_conv4(x, skip_connections[3], training=False)
        x = self.output_conv(x)
        pred = self.output_layer(x)

        return pred

    def test_step(self, inputs):
        x, y_true = inputs['image'], inputs['mask']

        # Forward pass
        y_pred = self.predict(x)

        dice = dice_score(y_true, y_pred)
        bfc = tf.keras.metrics.binary_focal_crossentropy(y_true, y_pred, gamma=2.0)

        return {'dice_score': dice, 'binary_focal_crossentropy': bfc}

    def train_step(self, inputs):
        x, y_true = inputs['image'], inputs['mask']

        # Forward pass
        with tf.GradientTape() as tape:
            y_pred = self.predict(x)
            loss = self.compiled_loss(y_true, y_pred)

        # Backward pass
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        dice = dice_score(y_true, y_pred)
        bfc = tf.keras.metrics.binary_focal_crossentropy(y_true, y_pred, gamma=2.0)

        return {'dice_score': dice, 'binary_focal_crossentropy': bfc}

    def compile(self, loss, optimizer='adam', learning_rate = 1e-2):
        if optimizer=='adam':
            opt = tf.keras.optimizers.Adam(lr = learning_rate)
        elif optimizer == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(lr = learning_rate)
        else:
            raise Exception(f'Base model {self.base_model_name} is not defined. \nChoose from ["adam","rmsprop"]')

        self.compile(
            optimizer = opt,
            loss = loss
        )

    def save(self):
        # Save the entire model to a HDF5 file
        self.save_weights(os.path.join(self.checkpoints_path, f'{self.name}.hdf5'))
        print('Model saved to:')
        print(os.path.join(self.checkpoints_path, f'{self.name}.hdf5'))

    def load(self):
        # Load the model weights from a HDF5 file
        self.load_weights(os.path.join(self.checkpoints_path, f'{self.name}.hdf5'))
        print('Model loaded from:')
        print(os.path.join(self.checkpoints_path, f'{self.name}.hdf5'))