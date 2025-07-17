# Heart-Disease-Prediction-using-Neural-Network-Project

I. Importing essential libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import os
# print(os.listdir())

import warnings
warnings.filterwarnings('ignore')

II. Importing and understanding our dataset
dataset = pd.read_csv("/content/heart.csv")

1.Shape of dataset
dataset.shape

2.Printing out a few columns
dataset.head(5)

