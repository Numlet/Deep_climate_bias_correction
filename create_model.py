#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 11:13:12 2019

@author: jvergara
"""

import sys
sys.path.append('/users/jvergara/python_code')
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
#tf.enable_eager_execution()

home_folder,store_folder,sc_folder=jle.Create_project_folders('BIAS_PRECIP_ML',sc=1)
switzerland_mask=np.load('/store/c2sm/pr04/jvergara/CONV_ON_OFF/swiss_mask.npy')

sess = tf.Session()
#%% =============================================================================
# Load data
# =============================================================================

model_data=np.load(store_folder+'COSMO_daily_cumulated_prec.npy')
#model_data=model_data[:1096]
obs_data=np.load(store_folder+'RdisaggH_daily_cumulated_prec.npy')[:model_data.shape[0],]
missing_data=obs_data<0
obs_data[missing_data]=0
model_data[missing_data]=0
bias=obs_data-model_data
#%% =============================================================================
# Plot current biases
# =============================================================================
ds_lat_lon=Dataset('/project/pr04/observations/meteoswiss/RdisaggH/RdisaggH_ch01r.swisscors_latlon.nc')

levels=np.linspace(-3,3,11)
jle.Quick_plot(bias.mean(axis=0),'Biases RdisaggH-COSMOpompa ',metadata_dataset=ds_lat_lon,levels=levels,extend='both',cmap=plt.cm.RdBu,cb_format="%1.2f",cb_label='mm/day')



#%% =============================================================================
# Split training and evaluation
# =============================================================================

mask=switzerland_mask
def reconstruct_array(array_flat,mask):
    indx=0
    r_array=np.zeros(mask.shape)
    for i in range(r_array.shape[0]):
        for j in range(r_array.shape[1]):
            if mask[i,j]:
                r_array[i,j]=np.nan
            else:
                r_array[i,j]=array_flat[indx]
                indx+=1
    return r_array
bias_flat=bias.mean(axis=0).flatten()
bias_flat_reduced=bias_flat[~switzerland_mask.flatten()]
bias_reconstructed=reconstruct_array(bias_flat_reduced,mask)
jle.Quick_plot(bias_reconstructed,'Biases RdisaggH-COSMOpompa ',metadata_dataset=ds_lat_lon,levels=levels,extend='both',cmap=plt.cm.RdBu,cb_format="%1.2f",cb_label='mm/day')
#bias_flat=bias.reshape((bias.shape[0],bias.shape[1]*bias.shape[2]))



bias_data_flat=bias.reshape((bias.shape[0],bias.shape[1]*bias.shape[2]))
model_data_flat=model_data.reshape((model_data.shape[0],model_data.shape[1]*model_data.shape[2]))
obs_data_flat=obs_data.reshape((obs_data.shape[0],obs_data.shape[1]*obs_data.shape[2]))

bias_data_flat=bias_data_flat[:,~switzerland_mask.flatten()]
model_data_flat=model_data_flat[:,~switzerland_mask.flatten()]
obs_data_flat=obs_data_flat[:,~switzerland_mask.flatten()]


cut=int(len(model_data_flat)*0.8)
input_tensor_train=model_data_flat[:cut,]
input_tensor_val=model_data_flat[cut:,]
target_tensor_train=bias_data_flat[:cut,]
target_tensor_val=bias_data_flat[cut:,]
target_tensor_val=bias_data_flat[cut:,]
obs_data_val=obs_data_flat[cut:,]
# input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(model_data_flat, bias_data_flat, test_size=0.2)




#%% =============================================================================
# Define model
# =============================================================================

if __name__=='__main__':
    #import tensorflow.contrib.eager as tfe
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1600,activation=tf.nn.relu,input_shape=(46718,)),
        tf.keras.layers.Dense(1600),
    #    tf.keras.layers.Dense(1600, activation=tf.nn.relu),
        tf.keras.layers.Dense(1600, activation=tf.nn.relu),
        tf.keras.layers.Dense(1600),
        tf.keras.layers.Dense(46718),
    ])
    
        
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    model.compile(loss='mse',
        optimizer=optimizer,
        metrics=['mae', 'mse'])
    #                metrics=)
    #                metrics=[tf.metrics.mean_absolute_error, 'mse'])
        
        
    #model.compile(optimizer=tf.train.GradientDescentOptimizer(0.1),
    #              loss='mean_squared_error')
    
    model.summary()
    
    
    
    #%% =============================================================================
    # Fit model
    # =============================================================================
    
    #tf.global_variables_initializer()
    
    history = model.fit(input_tensor_train,
                        target_tensor_train,
                        epochs=10,
    #                    batch_size=100,
                        validation_data=(input_tensor_val, target_tensor_val))#,
    #                    verbose=2)
    #sess = tf.Session()
    
    predictions=model(input_tensor_val)
    #predictions.eval(session=sess)
    predictions=predictions.numpy()
    #sess.run(predictions)
    #plt.plot(predictions)
    predictions_2D=reconstruct_array(predictions.mean(axis=0),switzerland_mask)
    jle.Quick_plot(predictions_2D,'Bias predictions RdisaggH-COSMOpompa ',metadata_dataset=ds_lat_lon,levels=levels,extend='both',cmap=plt.cm.RdBu,cb_format="%1.2f",cb_label='mm/day')
    
    
    model.save(store_folder+'my_model.h5')
    
    #%%
    
    new_model = tf.keras.models.load_model(store_folder+'my_model.h5')
    new_model.summary()
    
    #%%
    corrected_model=input_tensor_val+predictions
    
    
    jle.Quick_plot(reconstruct_array(predictions.mean(axis=0),switzerland_mask),'Bias predictions RdisaggH-COSMOpompa ',metadata_dataset=ds_lat_lon,levels=levels,extend='both',cmap=plt.cm.RdBu,cb_format="%1.2f",cb_label='mm/day')
    obs_data_val
    
    new_bias_flat=(obs_data_val-corrected_model).numpy()
    old_bias_flat=(target_tensor_val).numpy()
    
    new_bias=np.zeros((new_bias_flat.shape[0],switzerland_mask.shape[0],switzerland_mask.shape[1]))
    old_bias=np.zeros((old_bias_flat.shape[0],switzerland_mask.shape[0],switzerland_mask.shape[1]))
    for i in range(new_bias_flat.shape[0]):
        print(i)
        new_bias[i,]=reconstruct_array(new_bias_flat[i],switzerland_mask)
        old_bias[i,]=reconstruct_array(old_bias_flat[i],switzerland_mask)
    
    
    
    
    #corrected_model_2D 
    #jle.Quick_plot(bias.mean(axis=0),'Biases RdisaggH-COSMOpompa ',metadata_dataset=ds_lat_lon,levels=levels,extend='both',cmap=plt.cm.RdBu,cb_format="%1.2f",cb_label='mm/day')
    jle.Quick_plot(target_tensor_val.mean(axis=0),'Biases RdisaggH-COSMOpompa ',metadata_dataset=ds_lat_lon,levels=levels,extend='both',cmap=plt.cm.RdBu,cb_format="%1.2f",cb_label='mm/day')
    jle.Quick_plot(new_bias.mean(axis=0),'New biases RdisaggH-COSMOpompa ',metadata_dataset=ds_lat_lon,levels=levels,extend='both',cmap=plt.cm.RdBu,cb_format="%1.2f",cb_label='mm/day')
