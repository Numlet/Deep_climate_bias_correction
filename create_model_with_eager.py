#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 11:13:12 2019

@author: jvergara
"""
import matplotlib 
matplotlib.use('Agg')
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
if __name__=='__main__':tf.enable_eager_execution()

#home_folder,store_folder,sc_folder=jle.Create_project_folders('BIAS_PRECIP_ML',sc=1)
store_folder='/scratch/snx3000/jvergara/BIAS_PRECIP_ML/'
switzerland_mask=np.load(store_folder+'swiss_mask.npy')

model_name='3_layer_3_fields_1drop_l1'
#sess = tf.Session()
#%% =============================================================================
# Load data
# =============================================================================

model_data=np.load(store_folder+'COSMO_daily_cumulated_prec.npy')
model_data_T=np.load(store_folder+'COSMO_daily_mean_dd_T_2M.npy')
model_data_RH=np.load(store_folder+'COSMO_daily_mean_RELHUM_2M.npy')
#model_data_T=np.load(store_folder+'COSMO_daily_mean_ASWD_S.npy')
#model_data=model_data[:1096]
obs_data=np.load(store_folder+'RdisaggH_daily_cumulated_prec.npy')[:model_data.shape[0],]
missing_data=obs_data<0
obs_data[missing_data]=0
model_data[missing_data]=0
bias=obs_data-model_data
#%% =============================================================================
# Plot current biases
# =============================================================================
ds_lat_lon=Dataset(store_folder+'RdisaggH_ch01r.swisscors_latlon.nc')

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
model_data_T_flat=model_data.reshape((model_data.shape[0],model_data.shape[1]*model_data.shape[2]))
model_data_RH_flat=model_data.reshape((model_data.shape[0],model_data.shape[1]*model_data.shape[2]))
obs_data_flat=obs_data.reshape((obs_data.shape[0],obs_data.shape[1]*obs_data.shape[2]))

bias_data_flat=bias_data_flat[:,~switzerland_mask.flatten()]
model_data_flat=model_data_flat[:,~switzerland_mask.flatten()]
model_data_T_flat=model_data_T_flat[:,~switzerland_mask.flatten()]
model_data_RH_flat=model_data_RH_flat[:,~switzerland_mask.flatten()]
obs_data_flat=obs_data_flat[:,~switzerland_mask.flatten()]


cut=int(len(model_data_flat)*0.8)
model_data_T_train=model_data_T_flat[:cut,]
model_data_T_val=model_data_T_flat[cut:,]
model_data_RH_train=model_data_RH_flat[:cut,]
model_data_RH_val=model_data_RH_flat[cut:,]
input_tensor_train=model_data_flat[:cut,]
input_tensor_val=model_data_flat[cut:,]
target_tensor_train=bias_data_flat[:cut,]
target_tensor_val=bias_data_flat[cut:,]
#target_tensor_val=bias_data_flat[cut:,]
#obs_data_val=obs_data_flat[cut:,]
obs_data_val=obs_data_flat[cut:,]
obs_data_train=obs_data_flat[:cut,]




# input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(model_data_flat, bias_data_flat, test_size=0.2)

#%%
input_tensor_train=np.concatenate((input_tensor_train,model_data_T_train,model_data_RH_train),axis=1)
input_tensor_val=np.concatenate((input_tensor_val,model_data_T_val,model_data_RH_val),axis=1)


#%%
# =============================================================================
# Normalize
# =============================================================================

#Create means and std

means_input=input_tensor_train.mean(axis=0)
std_input=np.std(input_tensor_train,axis=0)

def Normalize_input(array):
    normalized=(array-means_input)/std_input
    return normalized
input_tensor_train_norm=np.apply_along_axis(Normalize_input,1,input_tensor_train)
input_tensor_val_norm=np.apply_along_axis(Normalize_input,1,input_tensor_val)

def DeNormalize_input(array_norm):
    array=(array_norm*std_input)+means_input
    return array

#%%
means_target=target_tensor_train.mean(axis=0)
std_target=np.std(target_tensor_train,axis=0)

def Normalize_target(array):
    normalized=(array-means_target)/std_target
    return normalized
target_tensor_train_norm=np.apply_along_axis(Normalize_target,1,target_tensor_train)
target_tensor_val_norm=np.apply_along_axis(Normalize_target,1,target_tensor_val)

def DeNormalize_target(array_norm):
    array=(array_norm*std_target)+means_target
    return array
target_tensor_train_norm_denorm=np.apply_along_axis(DeNormalize_target,1,target_tensor_train_norm)


#plt.hist(target_tensor_train[:,30])
#plt.hist(target_tensor_train_norm[30,:])
#plt.hist(target_tensor_train_norm_denorm[30,:])
#diffs=target_tensor_train_norm_denorm[30,:]-target_tensor_train[30,:]



#%% =============================================================================
# Define model
# =============================================================================
if __name__=='__main__':
    single_field_index=46718
    import tensorflow.contrib.eager as tfe
    '''
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(46718,activation=tf.nn.relu,input_shape=(single_field_index*2,)),
    #    tf.keras.layers.Dense(1600),
        tf.keras.layers.Dropout(0.5),
    #    tf.keras.layers.Dense(1600, activation=tf.nn.relu),
    #    tf.keras.layers.Dense(46718, activation=tf.nn.relu),
    #    tf.keras.layers.Dense(46718),
    #    tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(46718),
    ])
    '''
    '''
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1600,activation=tf.nn.relu,input_shape=(single_field_index*2,)),
        tf.keras.layers.Dense(3200,kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
    #    tf.keras.layers.Dense(1600, activation=tf.nn.relu),
        tf.keras.layers.Dense(6400,kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
    #    tf.keras.layers.Dense(46718),
    #    tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(46718),
    ])
    '''
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1600,activation=tf.nn.relu,input_shape=(single_field_index*3,)),
        tf.keras.layers.Dense(3200,kernel_regularizer=tf.keras.regularizers.l1(0.001)),
        tf.keras.layers.Dropout(0.5),
#        tf.keras.layers.Dense(1600, activation=tf.nn.relu),
        tf.keras.layers.Dense(6400,kernel_regularizer=tf.keras.regularizers.l1(0.001), activation=tf.nn.relu),
#        tf.keras.layers.Dropout(0.5),
    #    tf.keras.layers.Dense(46718),
    #    tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(46718),
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.02)
    optimizer=tf.train.RMSPropOptimizer(0.000001)
    def loss(model, inputs, targets):
      error = model(inputs) - targets
      return tf.reduce_mean(tf.square(error))
    model.compile(loss='mean_squared_error',
                    optimizer=optimizer)
    #                metrics=)
    #                metrics=[tf.metrics.mean_absolute_error, 'mse'])
        
        
    #model.compile(optimizer=tf.train.GradientDescentOptimizer(0.1),
    #              loss='mean_squared_error')
    
    model.summary()
    
    
    
    #%% =============================================================================
    # Fit model
    # =============================================================================
    
    #tf.global_variables_initializer()
    checkpoint_path = store_folder+"cp_"+model_name+"-{epoch:03d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                     save_weights_only=True,
                                                     verbose=1,period=50)
    from tensorflow.keras.callbacks import CSVLogger
    
    csv_logger = CSVLogger(store_folder+"model_history_"+model_name+".csv", append=True)
    
    history = model.fit(input_tensor_train,
                        target_tensor_train,
                        epochs=2500,
    #                    batch_size=100,
                        validation_data=(input_tensor_val, target_tensor_val),
                        callbacks = [cp_callback,csv_logger])#,
    #                    verbose=2)
    #sess = tf.Session()
    
    predictions=model(input_tensor_val)
    #predictions.eval(session=sess)
    predictions=predictions.numpy()
    #sess.run(predictions)
    #plt.plot(predictions)
    predictions_2D=reconstruct_array(predictions.mean(axis=0),switzerland_mask)
    jle.Quick_plot(predictions_2D,'Bias predictions RdisaggH-COSMOpompa ',metadata_dataset=ds_lat_lon,levels=levels,extend='both',cmap=plt.cm.RdBu,cb_format="%1.2f",cb_label='mm/day')
    
    #%%
    #corrected_model=input_tensor_val+predictions
    
    
    #jle.Quick_plot(reconstruct_array(predictions.mean(axis=0),switzerland_mask),'Bias predictions RdisaggH-COSMOpompa ',metadata_dataset=ds_lat_lon,levels=levels,extend='both',cmap=plt.cm.RdBu,cb_format="%1.2f",cb_label='mm/day')
    #obs_data_val
    
    
    model.save(store_folder+model_name+'.h5')
    
    '''
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
    '''
