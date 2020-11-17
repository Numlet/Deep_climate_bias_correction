#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:07:39 2019

@author: jvergara
"""


from create_model_randomized_data import *
import sys
sys.path.append('/users/jvergara/python_code')
import Jesuslib_ml as jlml
import Jesuslib_eth as jle
import numpy as np
import matplotlib.pyplot as plt
import glob
from netCDF4 import Dataset
import os
# TensorFlow and tf.keras
import tensorflow as tf
#from tensorflow import keras
from sklearn.model_selection import train_test_split
tf.enable_eager_execution()
#%%



rain_in=input_tensor_train[:,:46718]
rain_obs_in=np.copy(obs_data_train)



ordered_precip_train=np.sort(rain_in,axis=0)
ordered_precip_obs_train=np.sort(rain_obs_in,axis=0)


#plt.pcolormesh(ordered_precip)



rain_val=input_tensor_val[:,:46718]
rain_obs_val=np.copy(obs_data_val)

ordered_precip_val=np.sort(rain_val,axis=0)
ordered_precip_obs_val=np.sort(rain_obs_val,axis=0)



#%%


def find_nearest_vector_index(array, value):
    n = np.array([abs(i-value) for i in array])
    nindex=np.apply_along_axis(np.argmin,0,n)
    return nindex


igridbox=0
iday=0
def quantile_map(array):
    value=array[0]
    array=array[1:]
    indx=find_nearest_vector_index(array,value)
    new_value=ordered_precip_obs_train[indx,igridbox]
    return new_value

#%%

corrected_map_val=np.zeros_like(rain_val)

for iday in range(ordered_precip_val.shape[0]):
    print (iday)
    values=rain_val[iday,:]
    arrays=ordered_precip_train[:,:]
    array_to_feed=np.zeros((ordered_precip_train.shape[0]+1,ordered_precip_train.shape[1]))
    array_to_feed[0,:]=values
    array_to_feed[1:,:]=arrays
    day_array=np.apply_along_axis(quantile_map,0,array_to_feed)
    corrected_map_val[iday,:]=day_array
#    indx=find_nearest_vector_index(array,value)
#    new_value=ordered_precip_obs_train[indx,igridbox]
#    corrected_map_val[iday,igridbox]=value
np.save(store_folder+'quantile_mapped_precip_validation_with_apply_along_axis_randomized.npy',corrected_map_val)
#%%
corrected_map_val=np.zeros_like(rain_val)
for igridbox in range(ordered_precip_val.shape[1]):
    print(igridbox)
    for iday in range(ordered_precip_val.shape[0]):
        value=rain_val[iday,igridbox]
        array=ordered_precip_train[:,igridbox]
        indx=find_nearest_vector_index(array,value)
        new_value=ordered_precip_obs_train[indx,igridbox]
        corrected_map_val[iday,igridbox]=new_value

np.save(store_folder+'quantile_mapped_precip_validation_randomized.npy',corrected_map_val)