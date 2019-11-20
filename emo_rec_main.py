from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils


mat = loadmat('dataset_m.mat')
mdata = mat['dataset_m']

data = pd.DataFrame(mdata, dtype=mdata.dtype)
labels = np_utils.to_categorical(data[29])

model = Sequential()
