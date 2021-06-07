import librosa
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import h5py
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
f = h5py.File("s100_data.hdf5", "w")
import os ,sys
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
np.set_printoptions(threshold=sys.maxsize)
k = 20 # target dimension(s)
pca = PCA(k) # Create a new PCA instance

# generating data and saving into data file 
eeg_data=pd.read_hdf('for_mfcc.h5')

range_=int(eeg_data.shape[0]/4096)
start=0
for i in tqdm(range (range_)):
    sample_data=eeg_data[start:start+4096]
    name=str(sample_data['target'][start])+str(i)
  


    #y, sr = librosa.load(eeg_data['signal'])
    final_mcff=(librosa.feature.mfcc(y=np.array(sample_data['signal']), sr=128))
  
    pca_data=pca.fit_transform(final_mcff)


    dset = f.create_dataset(name,data= np.array(pca_data))
    start=start+4096
 


print(len(f.keys()))