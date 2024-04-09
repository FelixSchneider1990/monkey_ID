# Example usage:
cfg = Configuration()  # configuration class
ml = MachineLearning()  # machine learning class
np, sns, plt, torch, tf, keras, progressbar, os, cv2, pd, random, Path = cfg.import_packages()

IMG_path = cfg.set_directory('/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/data_analysis/data/')
MODEL_path = cfg.set_directory('/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/data_analysis/CNN_models/')
DATA_path = cfg.set_directory('/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/data_analysis/Pic_Labels/')
DF_path = cfg.set_directory('/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/data_analysis/dataframes/')

IMG_file_name = 'alwCla_20211222_084714_00005.jpg'
new_array, img_size = ml.prepare_img(os.path.join(IMG_path, IMG_file_name))

param = ml.define_model_params() # NEEDS INPUT TUPLE
model = ml.define_model_keras(param, img_size, 2)

DF_file_name = 'mID_stage2_TrialsDataFrame_20220607.csv'
df = ml.load_existing_dataframe(os.path.join(DF_path, DF_file_name))