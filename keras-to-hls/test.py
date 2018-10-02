import os
from optparse import OptionParser
from keras.models import load_model, Model
from keras.models import model_from_json
from argparse import ArgumentParser
from keras import backend as K
import numpy as np
import h5py
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import yaml
import math


json_string = open('fromSergo/model_Run3_2outputs.5.json', 'r').read()
model = model_from_json(json_string)
model.load_weights('fromSergo/model_Run3_2outputs_weights.5.h5')

#x_image_test = np.ones(80) #what is wrong with this??
#x_image_test = a = np.matrix('1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1')
x_image_test = a = np.matrix('10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10')

print(x_image_test.shape)
print(x_image_test)
print(model.predict(x_image_test))
