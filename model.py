import numpy as np
import keras
import csv
import cv2
import os
from tqdm import tqdm

class steering_model(object):

    def __init__(self):
        self.layers = [{'type': "dense", 'units': 1}]

    def preprocessing(self):
        self.model.add(keras.layers.Flatten(input_shape=(160, 320, 3)))

    def def_model(self):

        self.model = keras.models.Sequential()
        self.preprocessing()
        for layer in self.layers:
            if layer['type'] == 'dense':
                self.model.add(keras.layers.Dense(layer['units']))

        return self.model


def main(log_results, model_name):

    data_path = '../../../../../Volumes/home/Projekte/Data/CarND-Behavioral-Cloning-P3/data_recording'
    data = []

    with open(data_path + '/driving_log.csv') as csv_file:
        lines = csv.reader(csv_file)
        for line in lines:
            data.append(line)
        data = np.array(data)

    input_images_paths_old = data[:, 1]

    Label_data_str = data[:, 6]
    Input_image_data = []
    Label_data = []
    print("Extracting images from repository ...")
    for i in tqdm(range(data.shape[0])):

        path_new = os.path.join(data_path, 'IMG', input_images_paths_old[i].split('/')[-1])
        Input_image_data.append(cv2.imread(path_new))

        Label_data.append(float(Label_data_str[i]))

    Input_image_data = np.array(Input_image_data)
    Label_data = np.array(Label_data)


    model = steering_model().def_model()

    model.compile(loss='mse', optimizer='adam')
    model.fit(Input_image_data, Label_data, validation_split=0.2, shuffle=True, epochs=20, batch_size=32)

    if log_results == True:
        model.save(os.path.join('trained_models', 'model_{}.h5'.format(model_name)))


if __name__ == "__main__":

    model_name = "Dense"
    main(log_results=True, model_name=model_name)
