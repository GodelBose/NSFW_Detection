from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input, Dense, merge
from keras.applications.resnet50 import ResNet50
import numpy as np

class Detector(object):

    def __init__(self, img_shape, n_dense):
        '''Initializes the detector network:
        -----------

        img_shape: tuple
        A tuple denoting the image shape that the network is trained on. Should be of the form (height,width,channels)

        n_dense: int
        Number of units in the dense layers that are put on top of the pre trained ResNet.

        Returns:
        --------
        Initialized detector network.
        '''
        self.model = self.get_pre_resnet(img_shape, n_dense)

    def get_pre_resnet(self, input_shape, n_dense):
        '''Loads the pretrained Keras ResNet, adds 2 trainable dense layers and returns the compiled graph:
        -----------

        input_shape: tuple
        A tuple denoting the image shape that the network is trained on. Should be of the form (height,width,channels)

        n_dense: int
        Number of units in the dense layers that are put on top of the pre trained ResNet.

        Returns:
        --------
        Compiled detector network.
        '''
        base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape)
        x = base_model.output
        f_1 = Flatten()(x)
        n1 = BatchNormalization()(f_1)
        fc1 = Dense(n_dense)(n1)
        r1 = Activation('relu')(fc1)
        n2 = BatchNormalization()(r1)
        fc2 = Dense(n_dense)(n2)
        r2 = Activation('relu')(fc2)
        n3 = BatchNormalization()(r2)
        fc3 = Dense(2)(n3)
        final = Activation('softmax')(fc3)
        res_net = Model(input=base_model.input, output=final)

        for layer in base_model.layers:
            layer.trainable = False

        res_net.compile(optimizer='adam', loss='categorical_crossentropy',
                          metrics=['accuracy'])
        return res_net

    def train_on_batch(self, X,Y):
        '''Same method as in the Keras API:
        -----------

        X: np.array
        Numpy array containing the training data of shape (n_samples, height, width, channels)

        Y: np.array
        Corresponding labels for the training data.

        Returns:
        --------
        None
        '''
        self.model.train_on_batch(X.astype(np.float32), Y)

    def predict(self, X):
        '''Returns rounded predictions on the given data:
        -----------

        X: np.array
        Numpy array containing the data of shape (n_samples, height, width, channels)


        Returns:
        --------
        Prediction: np.array
        Numpy array containing the predictions with either 1 or 0
        '''
        return np.round(self.model.predict(X)[:,1])

    def save_weights(self, path):
        '''Same method as in the Keras API:
        -----------

        path: str
        Filename of the model to be saved
        Returns:
        --------
        None
        '''
        self.model.save_weights(path)

    def load_weights(self, path):
        '''Same method as in the Keras API:
        -----------

        path: str
        Filename of the model to be loaded
        Returns:
        --------
        None
        '''
        self.model.load_weights(path)
