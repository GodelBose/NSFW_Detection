from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input, Dense, merge
from keras.applications.resnet50 import ResNet50
import numpy as np

class Detector(object):

    def __init__(self, img_shape, n_dense):
        self.model = self.get_pre_resnet(img_shape, n_dense)

    def get_pre_resnet(self, input_shape, n_dense):
        base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224,224,3))
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
        self.model.train_on_batch(X.astype(np.float32), Y)

    def predict(self, X):
        return np.round(self.model.predict(X)[:,1])

    def save_weights(self, path):
        self.model.save_weights(path)
