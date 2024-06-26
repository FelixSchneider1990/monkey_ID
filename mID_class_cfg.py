class Configuration:
    """Configuration class"""

    @staticmethod
    def import_packages():
        # Import packages
        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt
        import torch
        import tensorflow as tf
        from tensorflow import keras
        from tqdm import tqdm
        import os
        import cv2
        import pandas as pd
        import random
        from pathlib import Path

        print("Packages imported successfully.")
        return np, sns, plt, torch, tf, keras, tqdm, os, cv2, pd, random, Path

    def set_directory(self, in_directory):
        # Call import_packages to get the packages
        np, sns, plt, torch, tf, keras, progressbar, os, cv2, pd, random, Path = self.import_packages()

        if os.path.isdir(in_directory):
            # Set directory - if it exists
            out_directory = in_directory
            print(f"Directory set to '{in_directory}'.")
        else:
            # Create directory or return error?
            out_directory = 'invalid'
            print("Please provide a valid directory.")
        return out_directory