import pandas as pd
import cv2
import random
import numpy as np
from tensorflow import keras
from tqdm import tqdm
import os

# Model's parameters (based on best model on Colab)
AvPool = 6
Conv_1 = 64
Conv_2 = 16
Conv_3 = 32
DenLay = 1024

IMG_path = '/Users/fschneider/Documents/MID_TestData/iggJos/20211220/'
DF_path = '/Users/fschneider/Documents/MID_TestData/'
Nsessions = 3
model_name = 'Felix_test'

print('Creating dataframe')
# Load Base Data Frame
csv_file = "{}{}".format(DF_path, 'BaseDataFrame.csv')
df = pd.read_csv(csv_file, index_col=0, low_memory=False)
categories = df['animal'].unique()
# Remove other monkeys from DF?

df['predict'] = 0
df['train'] = 0
df['result'] = 0
df['CNN_Npics'] = 0
df['CNN_Nsessions'] = Nsessions
df['CNN_type'] = model_name

df = df.reset_index(drop=True)

# ==========================================================================================
# Use data from the first three sessions to train the model
# Select subset of pictures for each session in the future
to_predict = df['session'].unique()[Nsessions:]
for i in to_predict:
    df.loc[df.session == i, 'predict'] = 1

# ==========================================================================================
# Select training pictures
n_samples = min(df[(df['predict'] == 0)]['animal'].value_counts())
df['CNN_Npics'] = n_samples
groups = df['group'].unique()
print('Select training pics')

for group in groups:
    print(group)
    animals = df[(df['predict'] == 0) & (df['group'] == group)]['animal'].unique()
    for m in animals:
        print(m)
        df.loc[(df['predict'] == 0) & (df['animal'] == m), 'train'] = 1
        n_samples = sum(df[(df['predict'] == 0) & (df['animal'] == m)]['train'])
        df.loc[df['animal'] == m, 'CNN_Npics'] = n_samples
        df.loc[df['animal'] == group, 'CNN_name'] = "{}{}{}".format(model_name, '_', group)

        non_test_list = df[(df['animal'] == m) & (df['predict'] == 0) & (df['group'] == group)] \
            .pic_name.to_list()
        selected = random.sample(non_test_list, n_samples)
        for r in tqdm(range(0, len(df))):
            if df.loc[r, 'pic_name'] in selected:
                df.loc[r, 'train'] = 1

save_name = "{}{}{}".format(DF_path, model_name, '.csv')
df.to_csv(save_name, sep=',')

def prepare(filepath):
    IMG_SIZE = 300
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_array = img_array / 255.0
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_array = img_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    return img_array

print('Import training pics')
IMG_SIZE = 300

# this doesn;t work
# new_array = prepare('/Users/fschneider/Documents/MID_TestData/iggJos/20211220/iggJos_20211221_092930_00360.jpg')
# same here
# '/Users/fschneider/Documents/MID_TestData/iggJos/20211220/iggJos_20211221_170834_03216.jpg'

for group in groups:
    print(group)
    X_train = []
    y_train = []

    animals = df[df['group'] == group]['animal'].unique()
    for m in animals:
        print(m)
        class_num = list(animals).index(m)
        all_elements = df[(df['animal'] == m) & (df['train'] == 1) & (df['group'] == group)].pic_name.to_list()

        for img in tqdm(all_elements):
            if not img.startswith('.'):
                if len(df[df['pic_name'] == img]):
                    actual_img_name = df[df['pic_name'] == img]['fileName'].values[0]
                    new_array = prepare(os.path.join(IMG_path, actual_img_name))
                    X_train.append(new_array)
                    y_train.append(class_num)

    X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_train = np.array(y_train)
    # this should be saved or returned

    # ==========================================================================================
    # Build the Convolutional Neural Network
    # model = keras.Sequential([
    #     # Average across AvPool pixels
    #     keras.layers.AveragePooling2D(AvPool, 3, input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    #
    #     # Go through Two hidden convolutional layers
    #     keras.layers.Conv2D(Conv_1, 3, activation='relu'),
    #     keras.layers.Conv2D(Conv_2, 3, activation='relu'),
    #     keras.layers.Conv2D(Conv_3, 3, activation='relu'),
    #
    #     # Pool across hidden layer 2 neurons
    #     keras.layers.MaxPool2D(2, 2),
    #     keras.layers.Dropout(0.5),
    #     keras.layers.Flatten(),
    #
    #     # Go through another hidden Layer
    #     keras.layers.Dense(DenLay, activation='relu'),
    #
    #     # Output layer
    #     keras.layers.Dense(len(animals), activation='softmax')
    # ])
    #
    # model.compile(optimizer='adam',
    #               loss=keras.losses.SparseCategoricalCrossentropy(),
    #               metrics=['accuracy'])
    #
    # model.fit(X_train, y_train,
    #           epochs=10,
    #           batch_size=32)
    #
    # NAME = "{}{}{}{}{}".format(MODEL_path, model_name, '_', group, '.model')
    # model.save(NAME)
    #
    # tf.keras.backend.clear_session()
    # gc.collect()