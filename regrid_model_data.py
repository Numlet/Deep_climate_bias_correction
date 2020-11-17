#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:43:33 2019

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

home_folder,store_folder,sc_folder=jle.Create_project_folders('BIAS_PRECIP_ML',sc=1)
model_data_folder=sc_folder

years=['2004','2005','2006','2007','2008']#,'2009','2010']

native_grid_file='/store/c2sm/pr04/jvergara/postprocessing_data/CLM_lm_f_grid_2.txt'
target_grid_file='/store/c2sm/pr04/jvergara/RdisaggH_grid.txt'

def Regrid(file,file_output_with_path,native_grid_file=native_grid_file,target_grid_file=target_grid_file,jump_past_files=0):
    do=1
    file_name_output_no_vcoord=file_output_with_path[:-3]+'_no_vcoord.nc'
#    file_name_output_no_vcoord=file
    if os.path.isfile(file_output_with_path) and jump_past_files:
        do=0
        return 0
    if do:
        if not os.path.exists('weights'):
            a=os.system('cdo delname,vcoord %s %s'%(file,file_name_output_no_vcoord))
            a=os.system('cdo genycon,%s -setgrid,%s %s weights'%(target_grid_file,native_grid_file,file_name_output_no_vcoord))
#            a=os.system('rm -f %s'%(file_name_output_no_vcoord))
        a=os.system('cdo delname,vcoord %s %s'%(file,file_name_output_no_vcoord))
        a=os.system('cdo remap,%s,weights -setgrid,%s %s %s'%(target_grid_file,native_grid_file,file_name_output_no_vcoord,file_output_with_path))
        os.system('rm -f %s'%(file_name_output_no_vcoord))
        return a



for year in years:
    print(year)
    year_folder='/project/pr04/davidle/results_clim/lm_f/1h/'+year+'/'
    files=glob.glob(year_folder+'*')
    for file in files:
        file_name=file.split('/')[-1]
        file_name_output=file_name[:-3]+'_regrided.nc'
        file_output_with_path=model_data_folder+file_name_output

        print(file_name)
        Regrid(file,file_output_with_path)
