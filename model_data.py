#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:41:29 2019

@author: jvergara
"""

import sys
sys.path.append('/users/jvergara/python_code')
import Jesuslib_eth as jle
import numpy as np
import matplotlib.pyplot as plt
import glob
from netCDF4 import Dataset


switzerland_mask=np.load('/store/c2sm/pr04/jvergara/CONV_ON_OFF/swiss_mask.npy')


home_folder,store_folder,sc_folder=jle.Create_project_folders('BIAS_PRECIP_ML',sc=1)



years=['2004','2005','2006','2007','2008']#,'2009','2010']

days=0
indx=0
sample_size=1827
aggregated_precip=np.zeros((sample_size,240, 370))

mean_RELHUM_2M=np.zeros((sample_size,240, 370))
mean_T_2M=np.zeros((sample_size,240, 370))
mean_CLCT=np.zeros((sample_size,240, 370))
mean_PS=np.zeros((sample_size,240, 370))
mean_ASWD_S=np.zeros((sample_size,240, 370))
mean_V_10M=np.zeros((sample_size,240, 370))
mean_U_10M=np.zeros((sample_size,240, 370))
#aggregated_flat=np.zeros((2557,46718))
for year in years:
    print(year)
    for month in jle.months_number_str:
        print(month)
        files=np.sort(glob.glob(sc_folder+'lffd'+year+month+'*'))
        print (len(files))
#        days+=len(files)
        days_in_month=len(files)/24
        
        month_long_prec=np.zeros((len(files),240, 370))
        ml_mean_RELHUM_2M=np.zeros((len(files),240, 370))
        ml_mean_T_2M=np.zeros((len(files),240, 370))
        ml_mean_CLCT=np.zeros((len(files),240, 370))
        ml_mean_PS=np.zeros((len(files),240, 370))
        ml_mean_ASWD_S=np.zeros((len(files),240, 370))
        ml_mean_V_10M=np.zeros((len(files),240, 370))
        ml_mean_U_10M=np.zeros((len(files),240, 370))
        for i in range(len(files)):
            file=files[i]
            print(file)
            ds=Dataset(file)
            
            month_long_prec[i,]=ds.variables['TOT_PREC'][0,]
            ml_mean_RELHUM_2M[i,]=ds.variables["RELHUM_2M"][0,]
            ml_mean_T_2M[i,]=ds.variables["T_2M"][0,]
            ml_mean_CLCT[i,]=ds.variables["CLCT"][0,]
            ml_mean_PS[i,]=ds.variables["PS"][0,]
            ml_mean_ASWD_S[i,]=ds.variables["ASWD_S"][0,]
            ml_mean_V_10M[i,]=ds.variables["V_10M"][0,]
            ml_mean_U_10M[i,]=ds.variables["U_10M"][0,]

        for i in range(int(days_in_month)):
            day_data_prec=month_long_prec[i*24:(i+1)*24]
            dd_RELHUM_2M=ml_mean_RELHUM_2M[i*24:(i+1)*24]
            dd_T_2M=ml_mean_T_2M[i*24:(i+1)*24]
            dd_CLCT=ml_mean_CLCT[i*24:(i+1)*24]
            dd_PS=ml_mean_PS[i*24:(i+1)*24]
            dd_ASWD_S=ml_mean_ASWD_S[i*24:(i+1)*24]
            dd_V_10M=ml_mean_V_10M[i*24:(i+1)*24]
            dd_U_10M=ml_mean_U_10M[i*24:(i+1)*24]
            
            
            
            day_data_prec[:,switzerland_mask]=np.nan
            dd_RELHUM_2M[:,switzerland_mask]=np.nan
            dd_T_2M[:,switzerland_mask]=np.nan
            dd_CLCT[:,switzerland_mask]=np.nan
            dd_PS[:,switzerland_mask]=np.nan
            dd_ASWD_S[:,switzerland_mask]=np.nan
            dd_V_10M[:,switzerland_mask]=np.nan
            dd_U_10M[:,switzerland_mask]=np.nan
            
            
            
            
            
            day_data_prec=day_data_prec.sum(axis=0)
            dd_RELHUM_2M=dd_RELHUM_2M.mean(axis=0)
            dd_T_2M=dd_T_2M.mean(axis=0)
            dd_CLCT=dd_CLCT.mean(axis=0)
            dd_PS=dd_PS.mean(axis=0)
            dd_ASWD_S=dd_ASWD_S.mean(axis=0)
            dd_V_10M=dd_V_10M.mean(axis=0)
            dd_U_10M=dd_U_10M.mean(axis=0)
            
            aggregated_precip[indx,]=day_data_prec
            
            mean_RELHUM_2M[indx,]=dd_RELHUM_2M
            mean_T_2M[indx,]=dd_T_2M
            mean_CLCT[indx,]=dd_CLCT
            mean_PS[indx,]=dd_PS
            mean_ASWD_S[indx,]=dd_ASWD_S
            mean_V_10M[indx,]=dd_V_10M
            mean_U_10M[indx,]=dd_U_10M
            
            indx+=1




np.save(store_folder+'COSMO_daily_cumulated_prec',aggregated_precip)
np.save(store_folder+'COSMO_daily_mean_RELHUM_2M',mean_RELHUM_2M)
np.save(store_folder+'COSMO_daily_mean_dd_T_2M',mean_T_2M)
np.save(store_folder+'COSMO_daily_mean_CLCT',mean_CLCT)
np.save(store_folder+'COSMO_daily_mean_PS',mean_PS)
np.save(store_folder+'COSMO_daily_mean_ASWD_S',mean_ASWD_S)
np.save(store_folder+'COSMO_daily_mean_V_10M',mean_V_10M)
np.save(store_folder+'COSMO_daily_mean_U_10M',mean_U_10M)



#plt.imshow(ds.variables["ASWD_S"][0,])
