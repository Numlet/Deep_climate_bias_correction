#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:18:39 2019

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
tf.enable_eager_execution()

home_folder,store_folder,sc_folder=jle.Create_project_folders('BIAS_PRECIP_ML',sc=1)
store_folder='/scratch/snx3000/jvergara/BIAS_PRECIP_ML/'
switzerland_mask=np.load(store_folder+'swiss_mask.npy')
#sess = tf.Session()
#%% =============================================================================
# Load data
# =============================================================================

model_data=np.load(store_folder+'COSMO_daily_mean_dd_T_2M.npy')
#model_data=model_data[:1096]
obs_data=np.load(store_folder+'TabsD_daily.npy')[:model_data.shape[0],]+273.15
missing_data=obs_data<0
obs_data[missing_data]=0
model_data[missing_data]=0
bias=model_data-obs_data
#%% =============================================================================
# Plot current biases
# =============================================================================
ds_lat_lon=Dataset(store_folder+'RdisaggH_ch01r.swisscors_latlon.nc')

levels=np.linspace(-3,3,11)
jle.Quick_plot(bias.mean(axis=0),'Biases COSMOpompa-Obs ',metadata_dataset=ds_lat_lon,levels=levels,extend='both',cmap=plt.cm.RdBu_r,cb_format="%1.2f",cb_label='K')



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
jle.Quick_plot(bias_reconstructed,'Biases COSMOpompa-Obs ',metadata_dataset=ds_lat_lon,levels=levels,extend='both',cmap=plt.cm.RdBu,cb_format="%1.2f",cb_label='mm/day')
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
obs_data_train=obs_data_flat[:cut,]
#%%


from create_model_T2M_and_Prec import *
import Jesuslib_ml as jlml
plt.figure()
jlml.plot_history('/scratch/snx3000/jvergara/BIAS_PRECIP_ML/model_history_'+model_name+'.csv')
plt.savefig(home_folder+"model_history")
levels_mae=np.linspace(0,5,11)

model = tf.keras.models.load_model(store_folder+model_name+'.h5')
model.summary()
cmap_MAE=plt.cm.rainbow
predictions=model(input_tensor_val)
predictions=predictions.numpy()
#predictions_2D=reconstruct_array(predictions.mean(axis=0),switzerland_mask)
new_map=input_tensor_val[:,46718:46718*2]-predictions
#new_map[new_map<0]=0

corrected_bias_val=new_map-obs_data_T_val
corrected_ae_val=np.absolute((new_map-obs_data_T_val))
plt.figure(figsize=(15,15))
#jle.Quick_plot,,metadata_dataset=ds_lat_lon,levels=levels,extend='both',cmap=plt.cm.RdBu,cb_format="%1.2f",cb_label='mm/day')
plt.subplot(221)
jle.Quick_plot(reconstruct_array((input_tensor_val[:,46718:46718*2]-obs_data_T_val).mean(axis=0),switzerland_mask),'Real Bias validation COSMOpompa-Obs ',
               metadata_dataset=ds_lat_lon,levels=levels,extend='both',cmap=plt.cm.RdBu_r,cb_format="%1.2f",cb_label='K',new_fig=0)
plt.subplot(222)
jle.Quick_plot(reconstruct_array((np.absolute(input_tensor_val[:,46718:46718*2]-obs_data_T_val)).mean(axis=0),switzerland_mask),'Real MAE validation COSMOpompa-Obs ',
               metadata_dataset=ds_lat_lon,levels=levels_mae,extend='max',cmap=cmap_MAE,cb_format="%1.2f",cb_label='K',new_fig=0)
plt.subplot(223)
jle.Quick_plot(reconstruct_array((corrected_bias_val).mean(axis=0),switzerland_mask),'Corrected Bias validation using ML COSMOpompa-Obs ',
               metadata_dataset=ds_lat_lon,levels=levels,extend='both',cmap=plt.cm.RdBu_r,cb_format="%1.2f",cb_label='K',new_fig=0)
plt.subplot(224)
jle.Quick_plot(reconstruct_array((corrected_ae_val).mean(axis=0),switzerland_mask),'Corrected MAE validation using ML COSMOpompa-Obs ',
               metadata_dataset=ds_lat_lon,levels=levels_mae,extend='max',cmap=cmap_MAE,cb_format="%1.2f",cb_label='K',new_fig=0)
plt.savefig(home_folder+"ML_validation_grid_T")
#%%
MAE_change=reconstruct_array((corrected_ae_val).mean(axis=0),switzerland_mask)-reconstruct_array((np.absolute(input_tensor_val[:,:46718]-obs_data_T_val)).mean(axis=0),switzerland_mask)
plt.figure(figsize=(8,8))
jle.Quick_plot(MAE_change,
               'Change in MAE Corrected-True validation using ML %1.3f'%np.nanmean(MAE_change),
               metadata_dataset=ds_lat_lon,levels=levels,extend='both',cmap=plt.cm.RdBu_r,cb_format="%1.2f",cb_label='K',new_fig=0)
plt.savefig(home_folder+"ML_MAE_change_validation_T")

#%%


predictions=model(input_tensor_train)
predictions=predictions.numpy()
#predictions_2D=reconstruct_array(predictions.mean(axis=0),switzerland_mask)
new_map=input_tensor_train[:,46718:46718*2]-predictions
#new_map[new_map<0]=0
corrected_bias_train=new_map-obs_data_T_train
corrected_ae_train=np.absolute((new_map-obs_data_T_train))
plt.figure(figsize=(15,15))
#jle.Quick_plot,,metadata_dataset=ds_lat_lon,levels=levels,extend='both',cmap=plt.cm.RdBu,cb_format="%1.2f",cb_label='mm/day')
plt.subplot(221)
jle.Quick_plot(reconstruct_array((input_tensor_train[:,:46718]-obs_data_train).mean(axis=0),switzerland_mask),'Real Bias training COSMOpompa-Obs ',
               metadata_dataset=ds_lat_lon,levels=levels,extend='both',cmap=plt.cm.RdBu,cb_format="%1.2f",cb_label='K',new_fig=0)
plt.subplot(222)
jle.Quick_plot(reconstruct_array((np.absolute(input_tensor_train[:,:46718]-obs_data_train)).mean(axis=0),switzerland_mask),'Real MAE trainign COSMOpompa-Obs ',
               metadata_dataset=ds_lat_lon,levels=levels_mae,extend='max',cmap=cmap_MAE,cb_format="%1.2f",cb_label='K',new_fig=0)
plt.subplot(223)
jle.Quick_plot(reconstruct_array((corrected_bias_train).mean(axis=0),switzerland_mask),'Corrected Bias training using ML COSMOpompa-Obs',
               metadata_dataset=ds_lat_lon,levels=levels,extend='both',cmap=plt.cm.RdBu,cb_format="%1.2f",cb_label='K',new_fig=0)

plt.subplot(224)
jle.Quick_plot(reconstruct_array((corrected_ae_train).mean(axis=0),switzerland_mask),'Corrected MAE training using ML COSMOpompa-Obs ',
               metadata_dataset=ds_lat_lon,levels=levels_mae,extend='max',cmap=cmap_MAE,cb_format="%1.2f",cb_label='K',new_fig=0)
plt.savefig(home_folder+"ML_train_grid_T")

#%%
plt.figure(figsize=(8,8))
MAE_change=reconstruct_array((corrected_ae_train).mean(axis=0),switzerland_mask)-reconstruct_array((np.absolute(input_tensor_train[:,:46718]-obs_data_train)).mean(axis=0),switzerland_mask)
jle.Quick_plot(MAE_change,
               'Change in MAE Corrected-True training using ML %1.3f'%np.nanmean(MAE_change),
               metadata_dataset=ds_lat_lon,levels=levels,extend='both',cmap=plt.cm.RdBu_r,cb_format="%1.2f",cb_label='mm/day',new_fig=0)
plt.savefig(home_folder+"ML_MAE_change_train_T")


#%%

#%%
corrected_map_val_QM=np.load(store_folder+'QM_temp.npy')[1461:,:]
#corrected_map_val_QM=np.roll(np.load(store_folder+'QM_prec.npy'),5000,axis=1)[1461:,:]
print(corrected_map_val_QM[0,0])
bias_QM_val=corrected_map_val_QM-obs_data_T_val
bias_QM_val=reconstruct_array(bias_QM_val.mean(axis=0),switzerland_mask)

MAE_QM_val=np.abs(corrected_map_val_QM-obs_data_T_val)
MAE_QM_val=reconstruct_array(MAE_QM_val.mean(axis=0),switzerland_mask)

#corrected_map_val_QM=np.load(store_folder+'quantile_mapped_precip_validation_with_apply_along_axis.npy')

#bias_QM_val=corrected_map_val_QM-obs_data_val
#bias_QM_val=reconstruct_array(bias_QM_val.mean(axis=0),switzerland_mask)

#MAE_QM_val=np.abs(corrected_map_val_QM-obs_data_val)
#MAE_QM_val=reconstruct_array(MAE_QM_val.mean(axis=0),switzerland_mask)

plt.figure(figsize=(15,15))
#jle.Quick_plot,,metadata_dataset=ds_lat_lon,levels=levels,extend='both',cmap=plt.cm.RdBu,cb_format="%1.2f",cb_label='mm/day')
plt.subplot(221)
jle.Quick_plot(reconstruct_array((input_tensor_val[:,46718:46718*2]-obs_data_T_val).mean(axis=0),switzerland_mask),'Real Bias validation RdisaggH-COSMOpompa ',
               metadata_dataset=ds_lat_lon,levels=levels,extend='both',cmap=plt.cm.RdBu_r,cb_format="%1.2f",cb_label='mm/day',new_fig=0)
plt.subplot(222)
jle.Quick_plot(reconstruct_array((np.absolute(input_tensor_val[:,46718:46718*2]-obs_data_T_val)).mean(axis=0),switzerland_mask),'Real MAE validation RdisaggH-COSMOpompa ',
               metadata_dataset=ds_lat_lon,levels=levels_mae,extend='max',cmap=cmap_MAE,cb_format="%1.2f",cb_label='mm/day',new_fig=0)
plt.subplot(223)
jle.Quick_plot(bias_QM_val,'Corrected Bias validation using QM RdisaggH-COSMOpompa ',
               metadata_dataset=ds_lat_lon,levels=levels,extend='both',cmap=plt.cm.RdBu,cb_format="%1.2f",cb_label='mm/day',new_fig=0)
plt.subplot(224)
jle.Quick_plot(MAE_QM_val,'Corrected MAE validation using QM RdisaggH-COSMOpompa ',
               metadata_dataset=ds_lat_lon,levels=levels_mae,extend='max',cmap=cmap_MAE,cb_format="%1.2f",cb_label='mm/day',new_fig=0)
plt.savefig(home_folder+"QM_validation_grid_T")

#%%
'''
plt.figure(figsize=(8,8))
MAE_change_QM=MAE_QM_val-reconstruct_array((np.absolute(input_tensor_val[:,:46718]-obs_data_val)).mean(axis=0),switzerland_mask)
jle.Quick_plot(MAE_change_QM,
               'Change in MAE Corrected-True validation using QM %1.3f'%np.nanmean(MAE_change_QM),
               metadata_dataset=ds_lat_lon,levels=levels,extend='both',cmap=plt.cm.RdBu_r,cb_format="%1.2f",cb_label='mm/day',new_fig=0)
plt.savefig(home_folder+"QM_MAE_change_validation")
'''