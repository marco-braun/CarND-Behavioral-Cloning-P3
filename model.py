import argparse
import cv2
import csv
import keras
from keras.callbacks import CSVLogger
import numpy as np
import os
from tqdm import tqdm

class steering_model(object):

    def __init__(self):
        self.layers = [{'type': "dense", 'units': 1}]

    def preprocessing_flatten(self):
        self.model.add(keras.layers.Flatten(input_shape=(160, 320, 3)))

    def def_model_dense(self):

        self.model = keras.models.Sequential()
        self.preprocessing_flatten()
        for layer in self.layers:
            if layer['type'] == 'dense':
                self.model.add(keras.layers.Dense(layer['units']))

        return self.model

    def def_model_lenet(self):

        self.model = keras.Sequential()

        self.model.add(keras.layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu'))
        self.model.add(keras.layers.AveragePooling2D())

        self.model.add(keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
        self.model.add(keras.layers.AveragePooling2D())

        self.model.add(keras.layers.Flatten())

        self.model.add(keras.layers.Dense(units=120, activation='relu'))

        self.model.add(keras.layers.Dense(units=84, activation='relu'))

        self.model.add(keras.layers.Dense(units=1))

        return self.model


def main(log_results, model_name, data_path):
    callbacks = []
    data = []

    with open(data_path + '/driving_log.csv') as csv_file:
        lines = csv.reader(csv_file)
        for line in lines:
            data.append(line)
        data = np.array(data)

    input_images_paths_old = data[:, 1]

    Label_data_str = data[:, 3] #steering
    Input_image_data = []
    Label_data = []
    print("Extracting images from repository ...")
    for i in tqdm(range(data.shape[0])):

        path_new = os.path.join(data_path, 'IMG', input_images_paths_old[i].split('/')[-1])
        Input_image_data.append(cv2.imread(path_new))

        Label_data.append(float(Label_data_str[i]))
        #break

    Input_image_data = np.array(Input_image_data, dtype=np.float32)
    Label_data = np.array(Label_data)


    model = steering_model().def_model_dense()

    model.compile(loss='mse', optimizer='adam')
    if log_results==True:
        dir_name = "Log_training_" + model_name
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        callbacks.append(CSVLogger(os.path.join(dir_name, 'training_log.csv'), append=True, separator=';'))
    model.fit(Input_image_data, Label_data, validation_split=0.2, shuffle=True, epochs=20, batch_size=32, callbacks=callbacks)

    if log_results == True:
        model.save(os.path.join(dir_name, 'model_{}.h5'.format(model_name)))




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Setup")
    parser.add_argument(
        'data_path',
        type=str,
        help='Path to the data',
        nargs='?',
        default='../../../../../Volumes/home/Projekte/Data/CarND-Behavioral-Cloning-P3/data_recording'
    )
    args = parser.parse_args()

    model_name = "LeNet"
    main(log_results=True, model_name=model_name, data_path=args.data_path)
