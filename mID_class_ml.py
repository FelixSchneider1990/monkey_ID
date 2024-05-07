# Import relevant packages
import sys
sys.path.append('/Users/fschneider/Documents/GitHub/monkeyID/')

from mID_class_cfg import Configuration

# Example usage:
cfg = Configuration()  # configuration class

# Call import_packages method directly from Configuration class
packages = cfg.import_packages()

# Assign the returned package names to global variables
np, sns, plt, torch, tf, keras, progressbar, os, cv2, pd, random, Path = packages

# Define machine learning class
class MachineLearning:
    def define_model_params(self):
        # Model's parameters (based on best model on Colab)
        # Determine type and number of layers as well as neurons/layer here - use tuples
        param = {
            "avPool": 6,
            "conv1": 64,
            "conv2": 16,
            "conv3": 32,
            "denLay": 1024
        }
        return param

    def define_model_keras(self, param, img_size, num_animals):
        # Build the Convolutional Neural Network
        model = keras.Sequential([
            # Average across AvPool pixels
            keras.layers.AveragePooling2D(param["avPool"], 3, input_shape=(img_size, img_size, 1)),

            # Go through two hidden convolutional layers
            keras.layers.Conv2D(param["conv1"], 3, activation='relu'),
            keras.layers.Conv2D(param["conv2"], 3, activation='relu'),
            keras.layers.Conv2D(param["conv3"], 3, activation='relu'),

            # Pool across hidden layer 2 neurons
            keras.layers.MaxPool2D(2, 2),
            keras.layers.Dropout(0.5),
            keras.layers.Flatten(),

            # Go through another hidden Layer
            keras.layers.Dense(param["denLay"], activation='relu'),

            # Output layer
            keras.layers.Dense(num_animals, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        return model

    def train_model(self, model, X_train, y_train):
        # Train model with labelled images
        model.fit(X_train, y_train,
                  epochs=10,
                  batch_size=32)
        tf.keras.backend.clear_session()
        # gc.collect()
        print('Model training concluded.')

    def save_model(self, in_model, name_string):
        # Save model to file
        in_model.save(name_string)
        print('Model has been saved.')

    def load_existing_dataframe(self, save_name):
        df = pd.read_csv(save_name, index_col=0, low_memory=False)
        return df

    @staticmethod
    def prepare_img(filepath):
        # Import, normalise, resize and vectorise image
        img_size = 300
        img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img_array = img_array / 255.0
        img_array = cv2.resize(img_array, (img_size, img_size))
        img_array = img_array.reshape(-1, img_size, img_size, 1)
        return img_array, img_size

    def build_dataframe(self,DF_path):
       df = ''

        return df

    @staticmethod
    def build_training_set(self, df):
        groups = df['group'].unique()

        for iGroup in groups:
            X_train = []
            y_train = []

            animals = df[df['group'] == iGroup]['animal'].unique()
            for m in animals:
                class_num = list(animals).index(m)
                all_elements = df[(df['animal'] == m) & (df['train'] == 1) & (df['group'] == iGroup)] \
                    .pic_name.to_list()

                for img in progressbar(all_elements):
                    if not img.startswith('.'):
                        if len(df[df['pic_name'] == img]):
                            IMG_file_name = df[df['pic_name'] == img]['fileName'].values[0]
                            new_array, img_size = ml.prepare_img(os.path.join(IMG_path, IMG_file_name))

                            X_train.append(new_array)
                            y_train.append(class_num)

            X_train = np.array(X_train).reshape(-1, img_size, img_size, 1)  # this has been done already?!
            y_train = np.array(y_train)

        return X_train, y_train
