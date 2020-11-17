#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:53:17 2019

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
data_folder='/project/pr04/observations/meteoswiss/RdisaggH/'
ds=Dataset('/project/pr04/observations/meteoswiss/RdisaggH/RdisaggH_ch01r.swisscors_200606010100_200607010000.nc')
ds_lat_lon=Dataset('/project/pr04/observations/meteoswiss/RdisaggH/RdisaggH_ch01r.swisscors_latlon.nc')
temp_folder='/store/c2sm/pr04/jvergara/meteoswiss_data/grid/'
home_folder,store_folder=jle.Create_project_folders('BIAS_PRECIP_ML')
#%%

years=['2004','2005','2006','2007','2008']#,'2009','2010']

days=0
indx=0
aggregated_precip=np.zeros((1827,240, 370))
aggregated_temp=np.zeros((1827,240, 370))
#aggregated_flat=np.zeros((2557,46718))
itemp=0
for year in years:
    print(year)
    file_temp=glob.glob(temp_folder+'TabsD_ch01r.swisscors_'+year+'*')[0]
    print (file_temp)
    ds_temp=Dataset(file_temp)
    days_in_year=ds_temp.variables['TabsD'].shape[0]
    aggregated_temp[itemp:days_in_year+itemp,]=ds_temp.variables['TabsD'][:]
    itemp=itemp+days_in_year
    for month in jle.months_number_str:
        print(month)
        file=glob.glob(data_folder+'RdisaggH_ch01r.swisscors_'+year+month+'*')[0]
        print (file)
        ds=Dataset(file)
#jle.Quick_plot(ds.variables['RdisaggH'][5,],'RdisaggH',metadata_dataset=ds_lat_lon)
        days_in_month=len(ds.variables['time'])/24
        days=days+days_in_month
        for i in range(int(days_in_month)):
            day_data=ds.variables["RdisaggH"][i*24:(i+1)*24].data
            day_data[:,switzerland_mask]=np.nan
#            day_data[day_data<0]
            day_data=day_data.sum(axis=0)
            aggregated_precip[indx,]=day_data
            indx+=1


np.save(store_folder+'RdisaggH_daily_cumulated_prec',aggregated_precip)
np.save(store_folder+'TabsD_daily',aggregated_temp)

#%%





jle.Quick_plot(ds.variables['RdisaggH'][5,],'RdisaggH',metadata_dataset=ds_lat_lon)



#data[switzerland_mask]=np.nan
ds_model=Dataset('/scratch/snx3000/jvergara/cajon_de_sastre/BIAS_PRECIP_ML/lffd2004052908_regrided.nc')

data=ds_model.variables["TOT_PREC"][0,].data
jle.Quick_plot(data,'RdisaggH',metadata_dataset=ds_lat_lon)


plt.figure()
#plt.imshow(ds.variables['RdisaggH'][5,])
plt.show()
#plt.imshow(ds_model.variables["T_2M"][0,0,])
plt.show()
