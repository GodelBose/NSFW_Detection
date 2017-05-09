import alexnet
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.image import PatchExtractor
import os
from PIL import Image
import itertools
from sklearn.metrics import accuracy_score
import sys
from detector import Detector

if __name__=='__main__':
    # model and training parameters
    model_name = 'resnet50'
    img_channels = 3
    width = 224
    height = 224
    batch_size = 64
    load = False
    nb_epochs = 100
    nb_dense = 4000
    i = 0

    try:
        experiment_name = sys.argv[1]
        os.mkdir(experiment_name)
    except:
        raise Exception('Please provide a non existing experiment directory name.')
    # Path for training data
    normal_path = os.getcwd()+'/train/normal'
    nsfw_path = os.getcwd()+'/train/nsfw'
    # Path for validation/testing data
    normal_path_test = os.getcwd()+'/test/normal'
    nsfw_path_test = os.getcwd()+'/test/nsfw'
    losses = []
    test_losses =[]
    num_train_samples = len(os.listdir(nsfw_path)) + len(os.listdir(normal_path))
    num_test_samples = len(os.listdir(nsfw_path_test)) + len(os.listdir(normal_path_test))

    model = Detector((height, width, img_channels), nb_dense)
    train_datagen = ImageDataGenerator(
            featurewise_center=False,
            featurewise_std_normalization=False,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator()
    X_test = np.zeros((num_test_samples,height,width,img_channels))
    y_test = []
    for x_temp,y_temp in test_datagen.flow_from_directory(os.getcwd()+'/test', target_size=(height,width), classes=['nsfw', 'normal'], batch_size=1, save_to_dir=os.getcwd()+'/test_pics', save_prefix='gen_pic_'):
        X_test[i] = x_temp
        y_test.append(np.where(np.array(y_temp)==1)[1][0])
        i += 1
        if i == num_test_samples:
            break

    X_test = X_test[:i]
    y_test = y_test[:i]
    losses = []
    test_losses = []
    for e in range(nb_epochs):
        batch_num = 0
        for X_batch,y_batch in train_datagen.flow_from_directory(os.getcwd()+'/train', target_size=(height,width), classes=['nsfw', 'normal'], batch_size=batch_size, save_to_dir=os.getcwd()+'/train_pics', save_prefix='gen_pic_'):
            loss = model.train_on_batch(X_batch.astype(np.float32), y_batch)
            losses.append(loss)
            batch_num += 1
            # Store results after each epoch
            if batch_num*batch_size > num_train_samples:
                testing_loss = accuracy_score(np.array(y_test), model.predict(X_test.astype(np.float32)))
                test_losses.append(testing_loss)
                model.save_weights(experiment_name+'/weights_'+str(e))
                plt.plot(test_losses)
                plt.title('testing accuracy')
                plt.xlabel('epoch')
                plt.ylabel('accuracy')
                plt.savefig(experiment_name+"/test_losses "+str(e))
                plt.clf()
                break
