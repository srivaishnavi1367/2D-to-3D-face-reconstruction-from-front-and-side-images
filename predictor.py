import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope

from Networks.resfcn256_6 import resfcn256_6
from Networks import mobilenet_v2


class PosPrediction_6():
    def __init__(self, resolution_inp=256, resolution_op=256):
        # -- hyper settings
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.MaxPos = resolution_inp * 1.1

        # network type
        self.network = resfcn256_6(self.resolution_inp, self.resolution_op)

    def restore(self, model_path):
        self.network.load_weights(model_path)  # Load model weights properly in TF2

    def predict(self, image):
        pos = self.network(image[np.newaxis, :, :, :], training=False)  # Inference mode
        pos = np.squeeze(pos)
        return pos * self.MaxPos

    def predict_batch(self, images):
        pos = self.network(images, training=False)
        return pos * self.MaxPos


class MobilenetPosPredictor():
    def __init__(self, resolution_inp=256, resolution_op=256):
        # -- hyper settings
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.MaxPos = resolution_inp * 1.1  # this *1.1 is somewhat of a mystery..
        self.model = None

        # Set TensorFlow GPU memory growth (TF2 equivalent of `tf.Session(config)`)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

    def restore(self, model_path):
        with CustomObjectScope({'relu6': mobilenet_v2.relu6}):
            self.model = keras.models.load_model(model_path)

    def predict(self, image):
        x = image[np.newaxis, :, :, :]
        pos = self.model.predict(x)
        pos = np.squeeze(pos)
        return pos * self.MaxPos

    def predict_batch(self, images):
        raise NotImplementedError


class PosPrediction_6_keras():
    def __init__(self, resolution_inp=256, resolution_op=256):
        # -- hyper settings
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.MaxPos = resolution_inp * 1.1  # this *1.1 is somewhat of a mystery..
        self.model = None

        # Set TensorFlow GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

    def restore(self, model_path):
        self.model = keras.models.load_model(model_path)

    def predict(self, image):
        pos = self.model.predict(image[np.newaxis, :, :, :])
        pos = np.squeeze(pos)
        return pos * self.MaxPos

    def predict_batch(self, images):
        raise NotImplementedError
