import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization


def resBlock(x, num_outputs, kernel_size=4, stride=1, activation_fn=tf.nn.relu, normalizer_fn=None, scope=None):
    assert num_outputs % 2 == 0  # num_outputs must be divided by channel_factor (2 here)

    shortcut = x
    if stride != 1 or x.shape[-1] != num_outputs:
        shortcut = layers.Conv2D(filters=num_outputs, kernel_size=1, strides=stride, padding='same',
                                 activation=None, kernel_regularizer=l2(0.0002))(shortcut)

    x = layers.Conv2D(filters=num_outputs // 2, kernel_size=1, strides=1, padding='same',
                      kernel_regularizer=l2(0.0002))(x)
    x = layers.Conv2D(filters=num_outputs // 2, kernel_size=kernel_size, strides=stride, padding='same',
                      kernel_regularizer=l2(0.0002))(x)
    x = layers.Conv2D(filters=num_outputs, kernel_size=1, strides=1, activation=None, padding='same',
                      kernel_regularizer=l2(0.0002))(x)

    x = layers.Add()([x, shortcut])  # Skip connection
    if normalizer_fn:
        x = normalizer_fn(x)
    x = activation_fn(x)
    return x


class resfcn256_6(tf.keras.Model):
    def __init__(self, resolution_inp=256, resolution_op=256, channel=6, name='resfcn256'):
        super(resfcn256_6, self).__init__(name=name)
        self.channel = channel
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.normalizer_fn = BatchNormalization()

    def call(self, x, training=True):
        size = 16
        x = layers.Conv2D(filters=size, kernel_size=4, strides=1, padding='same',
                          kernel_regularizer=l2(0.0002))(x)

        x = resBlock(x, num_outputs=size * 2, kernel_size=4, stride=2, normalizer_fn=self.normalizer_fn)
        x = resBlock(x, num_outputs=size * 2, kernel_size=4, stride=1, normalizer_fn=self.normalizer_fn)
        x = resBlock(x, num_outputs=size * 4, kernel_size=4, stride=2, normalizer_fn=self.normalizer_fn)
        x = resBlock(x, num_outputs=size * 4, kernel_size=4, stride=1, normalizer_fn=self.normalizer_fn)
        x = resBlock(x, num_outputs=size * 8, kernel_size=4, stride=2, normalizer_fn=self.normalizer_fn)
        x = resBlock(x, num_outputs=size * 8, kernel_size=4, stride=1, normalizer_fn=self.normalizer_fn)
        x = resBlock(x, num_outputs=size * 16, kernel_size=4, stride=2, normalizer_fn=self.normalizer_fn)
        x = resBlock(x, num_outputs=size * 16, kernel_size=4, stride=1, normalizer_fn=self.normalizer_fn)
        x = resBlock(x, num_outputs=size * 32, kernel_size=4, stride=2, normalizer_fn=self.normalizer_fn)
        x = resBlock(x, num_outputs=size * 32, kernel_size=4, stride=1, normalizer_fn=self.normalizer_fn)

        # Upsampling (Deconvolution)
        x = layers.Conv2DTranspose(filters=size * 32, kernel_size=4, strides=1, padding='same',
                                   kernel_regularizer=l2(0.0002))(x)
        x = layers.Conv2DTranspose(filters=size * 16, kernel_size=4, strides=2, padding='same',
                                   kernel_regularizer=l2(0.0002))(x)
        x = layers.Conv2DTranspose(filters=size * 16, kernel_size=4, strides=1, padding='same',
                                   kernel_regularizer=l2(0.0002))(x)
        x = layers.Conv2DTranspose(filters=size * 16, kernel_size=4, strides=1, padding='same',
                                   kernel_regularizer=l2(0.0002))(x)

        x = layers.Conv2DTranspose(filters=size * 8, kernel_size=4, strides=2, padding='same',
                                   kernel_regularizer=l2(0.0002))(x)
        x = layers.Conv2DTranspose(filters=size * 8, kernel_size=4, strides=1, padding='same',
                                   kernel_regularizer=l2(0.0002))(x)
        x = layers.Conv2DTranspose(filters=size * 8, kernel_size=4, strides=1, padding='same',
                                   kernel_regularizer=l2(0.0002))(x)

        x = layers.Conv2DTranspose(filters=size * 4, kernel_size=4, strides=2, padding='same',
                                   kernel_regularizer=l2(0.0002))(x)
        x = layers.Conv2DTranspose(filters=size * 4, kernel_size=4, strides=1, padding='same',
                                   kernel_regularizer=l2(0.0002))(x)
        x = layers.Conv2DTranspose(filters=size * 4, kernel_size=4, strides=1, padding='same',
                                   kernel_regularizer=l2(0.0002))(x)

        x = layers.Conv2DTranspose(filters=size * 2, kernel_size=4, strides=2, padding='same',
                                   kernel_regularizer=l2(0.0002))(x)
        x = layers.Conv2DTranspose(filters=size * 2, kernel_size=4, strides=1, padding='same',
                                   kernel_regularizer=l2(0.0002))(x)
        x = layers.Conv2DTranspose(filters=size, kernel_size=4, strides=2, padding='same',
                                   kernel_regularizer=l2(0.0002))(x)
        x = layers.Conv2DTranspose(filters=size, kernel_size=4, strides=1, padding='same',
                                   kernel_regularizer=l2(0.0002))(x)

        x = layers.Conv2DTranspose(filters=3, kernel_size=4, strides=1, padding='same')(x)
        x = layers.Conv2DTranspose(filters=3, kernel_size=4, strides=1, padding='same')(x)
        pos = layers.Conv2DTranspose(filters=3, kernel_size=4, strides=1, activation='sigmoid', padding='same')(x)

        return pos

    @property
    def vars(self):
        return self.trainable_variables
