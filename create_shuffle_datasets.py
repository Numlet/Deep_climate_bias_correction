#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:21:06 2019

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




#COSMO_daily_mean_ASWD_S.npy
#COSMO_daily_mean_CLCT.npy
#COSMO_daily_mean_PS.npy
#COSMO_daily_mean_U_10M.npy
#COSMO_daily_mean_V_10M.npy


model_data=np.load(store_folder+'COSMO_daily_cumulated_prec.npy')
model_data_T=np.load(store_folder+'COSMO_daily_mean_dd_T_2M.npy')
model_data_RH=np.load(store_folder+'COSMO_daily_mean_RELHUM_2M.npy')
model_data_ASWD_S=np.load(store_folder+'COSMO_daily_mean_ASWD_S.npy')
model_data_CLCT=np.load(store_folder+'COSMO_daily_mean_CLCT.npy')
model_data_PS=np.load(store_folder+'COSMO_daily_mean_PS.npy')
model_data_U_10M=np.load(store_folder+'COSMO_daily_mean_U_10M.npy')
model_data_V_10M=np.load(store_folder+'COSMO_daily_mean_V_10M.npy')






obs_data=np.load(store_folder+'RdisaggH_daily_cumulated_prec.npy')[:model_data.shape[0],]


#%%
np.random.seed(13061991)
positions=np.arange(1827)
np.random.shuffle(positions)



random_model_data = model_data[positions,]
random_obs_data = obs_data[positions,]
random_model_data_T = model_data_T[positions,]
random_model_data_RH = model_data_RH[positions,]
random_model_data_ASWD_S = model_data_ASWD_S[positions,]
random_model_data_CLCT = model_data_CLCT[positions,]
random_model_data_PS = model_data_PS[positions,]
random_model_data_U_10M = model_data_U_10M[positions,]
random_model_data_V_10M = model_data_V_10M[positions,]



np.save(store_folder+"RdisaggH_daily_cumulated_prec_random.npy",random_obs_data)
np.save(store_folder+"COSMO_daily_cumulated_prec_random.npy",random_model_data)
np.save(store_folder+"COSMO_daily_mean_dd_T_2M_random.npy",random_model_data_T)
np.save(store_folder+"COSMO_daily_mean_RELHUM_2M_random.npy",random_model_data_RH)
np.save(store_folder+"COSMO_daily_mean_ASWD_S_random.npy",random_model_data_ASWD_S)
np.save(store_folder+"COSMO_daily_mean_CLCT_random.npy",random_model_data_CLCT)
np.save(store_folder+"COSMO_daily_mean_PS_random.npy",random_model_data_PS)
np.save(store_folder+"COSMO_daily_mean_U_10M_random.npy",random_model_data_U_10M)
np.save(store_folder+"COSMO_daily_mean_V_10M_random.npy",random_model_data_V_10M)
