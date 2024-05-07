import sys

sys.path.append('/Users/fschneider/Documents/GitHub/monkeyID/')

from mID_class_ml import MachineLearning
from mID_class_cfg import Configuration

# Example usage:
cfg = Configuration()  # configuration class
ml = MachineLearning()  # machine learning class

# Call import_packages method directly from Configuration class
packages = cfg.import_packages()

# Assign the returned package names to global variables
np, sns, plt, torch, tf, keras, tqdm, os, cv2, pd, random, Path = packages

IMG_path = cfg.set_directory('/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/data_analysis/data/')
MODEL_path = cfg.set_directory('/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/data_analysis/CNN_models/')
DATA_path = cfg.set_directory('/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/data_analysis/Pic_Labels/')
DF_path = cfg.set_directory('/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/data_analysis/dataframes/')

# IMG_file_name = os.path.join(IMG_path, 'alwCla_20211222_084714_00005.jpg')
# new_array, img_size = ml.prepare_img(IMG_file_name)

param = ml.define_model_params()  # NEEDS INPUT TUPLE
model = ml.define_model_keras(param, img_size, 2)

# Copy pics locally - change paths
# Build dataframe based on pictures

# Load existing dataframe
# Build training set
# Train ML model

DF_file_name = 'mID_stage2_TrialsDataFrame_20220607.csv'
df = ml.load_existing_dataframe(os.path.join(DF_path, DF_file_name))
