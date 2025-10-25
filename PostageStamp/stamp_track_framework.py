import matplotlib.pyplot as plt
import pyart
import numpy as np
import numpy.ma as ma
from metpy.units import check_units, concatenate, units
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from siphon.radarserver import RadarServer
#rs = RadarServer('http://thredds-aws.unidata.ucar.edu/thredds/radarServer/nexrad/level2/S3/')
#rs = RadarServer('http://thredds.ucar.edu/thredds/radarServer/nexrad/level2/IDD/')
from datetime import datetime, timedelta
#from siphon.cdmr import Dataset
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from shapely.geometry import polygon as sp
import pyproj 
import shapely.ops as ops
from shapely.ops import transform
from shapely.geometry.polygon import Polygon
from functools import partial
from shapely import geometry
import netCDF4
from scipy import ndimage as ndi
#from skimage.feature import peak_local_max
#from skimage import data, img_as_float
from pyproj import Geod
from metpy.calc import wind_direction, wind_speed, wind_components
import matplotlib.lines as mlines
import pandas as pd
import scipy.stats as stats
import csv
import pickle
from sklearn.ensemble import RandomForestClassifier
import nexradaws
import os
#from grid_section import gridding
#from grid_section_spin import gridding_spin
from grid_section_small import gridding_spin_fast
from kdp_section import kdp_genesis
from gradient_section import grad_mask
#from ungridded_section import quality_control
from ungridded_section_spin import quality_control_spin
#from stormid_section import storm_objects
#from stormid_section_xtrap import storm_objects_new
#from zdr_arc_section import zdrarc
# from zdr_arc_trendss import zdrarc
# from hail_section import hail_objects
# from zhh_section import zhh_objects
# from kdpfoot_section import kdp_objects
# from zdr_col_section import zdrcol
from rotation_small import get_rotation
# from rotation_matching_qc import rot_storm_matcher_qc
from RAP_Archive import get_RAP_data
from metpy.interpolate import interpolate_to_points
import threading
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
import onnxruntime as rt

def multi_case_algorithm_2020(year, month, day, hour, start_min, duration, calibration, station, st_lon, st_lat, Z0C1, FFD1):
    REFlev = [40]
    n=1
    GR_mins=5
    Z0C=3500
    f_l = 0
    #Set vector perpendicular to FFD Z gradient
    #Make lists for the stamps
    REF_stamps = []
    ZDR_stamps = []
    CC_stamps = []
    KDP_stamps = []
    lon_stamps = []
    lat_stamps = []
    postlons = []
    postlats = []
    ROT_stamps = []
    ZDR_D_stamps = []
    NZDR_stamps = []
    time_stamps = []
    azim_stamps = []
    VEL_stamps = []
    #storm_relative_dir = storm_relative_dir
    #Set storm motion
    #Bunkers_m = Bunkers_m

    #Here, set the initial time of the archived radar loop you want.
    #Our specified time
    dt0 = datetime(year,month, day, hour, start_min)
    dt = dt0
    station = station
    end_dt = dt + timedelta(hours=duration)

    #Set up nexrad interface
    conn = nexradaws.NexradAwsInterface()
    scans = conn.get_avail_scans_in_range(dt,end_dt,station)
    results = conn.download(scans, 'RadarFolder')

    for i,scan in enumerate(results.iter_success(),start=1):
    #Local file option:
        #Loop over all files in the dataset and pull out each 0.5 degree tilt for analysis
        try:
            radar1 = scan.open_pyart()
        except:
            print('bad radar file')
            continue
        #Local file option
        print('File Reading')

        #Calling ungridded_section; Pulling apart radar sweeps and creating ungridded data arrays
        [radar,radar_v,n,range_2d,last_height,rlons_h,rlats_h,ungrid_lons,ungrid_lats] = quality_control_spin(radar1,n,calibration)

        time_start = netCDF4.num2date(radar.time['data'][0], radar.time['units'])
        object_number=0.0
        month = time_start.month
        if month < 10:
            month = '0'+str(month)
        hour = time_start.hour
        if hour < 10:
            hour = '0'+str(hour)
        minute = time_start.minute
        if minute < 10:
            minute = '0'+str(minute)
        day = time_start.day
        if day < 10:
            day = '0'+str(day)
        time_beg = time_start - timedelta(minutes=0.1)
        time_end = time_start + timedelta(minutes=GR_mins)
        sec_beg = time_beg.second
        sec_end = time_end.second
        min_beg = time_beg.minute
        min_end = time_end.minute
        h_beg = time_beg.hour
        h_end = time_end.hour
        d_beg = time_beg.day
        d_end = time_end.day
        if sec_beg < 10:
            sec_beg = '0'+str(sec_beg)
        if sec_end < 10:
            sec_end = '0'+str(sec_end)
        if min_beg < 10:
            min_beg = '0'+str(min_beg)
        if min_end < 10:
            min_end = '0'+str(min_end)
        if h_beg < 10:
            h_beg = '0'+str(h_beg)
        if h_end < 10:
            h_end = '0'+str(h_end)
        if d_beg < 10:
            d_beg = '0'+str(d_beg)
        if d_end < 10:
            d_end = '0'+str(d_end)
            
        #Calling kdp_section; Using NWS method, creating ungridded, smoothed KDP field
        kdp_nwsdict = kdp_genesis(radar)

        #Add field to radar
        radar.add_field('KDP', kdp_nwsdict)
        kdp_ungridded_nws = radar.fields['KDP']['data']


        #Calling grid_section; Now let's grid the data on a ~250 m x 250 m grid
        [Zint,REF,KDP,CC,CC_c,CCall,ZDRmasked1,ZDRrmasked1,REFmasked,REFrmasked,KDPmasked,KDPrmasked,rlons,rlats,rlons_2d,rlats_2d,cenlat,cenlon,VEL, REFall, ZDRall, KDPall] = gridding_spin_fast(radar,radar_v,Z0C,st_lon,st_lat)  
            
        #Get RAP data for FFD angles
            
        if f_l == 0:
            try:
                ffd_lons, ffd_lats, ffd_Rangles, Z0C_arl1 = get_RAP_data(cenlon, cenlat, time_start)
    
                radar_z = radar.altitude['data']
                Z0C_arl = Z0C_arl1 - (radar_z*units('m'))
    
                ffd_grid = interpolate_to_points(np.asarray([np.ndarray.flatten(ffd_lons), np.ndarray.flatten(ffd_lats)]).T,
                                                np.ndarray.flatten(ffd_Rangles.magnitude-86),
                                                np.asarray([np.ndarray.flatten(rlons_2d), np.ndarray.flatten(rlats_2d)]).T)
                ffd_sgrid = ffd_grid.reshape(rlons_2d.shape)
                storm_relative_dir = ffd_sgrid
    
                Z0C_grid = interpolate_to_points(np.asarray([np.ndarray.flatten(ffd_lons), np.ndarray.flatten(ffd_lats)]).T,
                                  np.ndarray.flatten(Z0C_arl[0,:,:].magnitude),
                                  np.asarray([np.ndarray.flatten(rlons_2d), np.ndarray.flatten(rlats_2d)]).T)
                Z0C_sgrid = Z0C_grid.reshape(rlons_2d.shape)
                ZINT_sgrid = np.round(Z0C_sgrid/250)
    
                time_RAPscan = time_start
            except:
                Z0C_sgrid = np.copy(rlons_2d)
                Z0C_sgrid[:] = Z0C1
                ZINT_sgrid = np.round(Z0C_sgrid/250)
                time_RAPscan = time_start
                ffd_sgrid = np.copy(rlons_2d)
                ffd_sgrid[:] = FFD1
                storm_relative_dir = ffd_sgrid
                
        else:
            diff_seconds = (time_start-time_RAPscan).seconds
            if diff_seconds > 3600:
                try:
                    ffd_lons, ffd_lats, ffd_Rangles, Z0C_arl1 = get_RAP_data(cenlon, cenlat, time_start)
    
                    radar_z = radar.altitude['data']
                    Z0C_arl = Z0C_arl1 - (radar_z*units('m'))
    
                    ffd_grid = interpolate_to_points(np.asarray([np.ndarray.flatten(ffd_lons), np.ndarray.flatten(ffd_lats)]).T,
                                                    np.ndarray.flatten(ffd_Rangles.magnitude-86),
                                                    np.asarray([np.ndarray.flatten(rlons_2d), np.ndarray.flatten(rlats_2d)]).T)
                    ffd_sgrid = ffd_grid.reshape(rlons_2d.shape)
                    storm_relative_dir = ffd_sgrid
    
                    Z0C_grid = interpolate_to_points(np.asarray([np.ndarray.flatten(ffd_lons), np.ndarray.flatten(ffd_lats)]).T,
                                  np.ndarray.flatten(Z0C_arl[0,:,:].magnitude),
                                  np.asarray([np.ndarray.flatten(rlons_2d), np.ndarray.flatten(rlats_2d)]).T)
                    Z0C_sgrid = Z0C_grid.reshape(rlons_2d.shape)
                    ZINT_sgrid = np.round(Z0C_sgrid/250)
    
                    time_RAPscan = time_start
                except:
                    Z0C_sgrid = np.copy(rlons_2d)
                    Z0C_sgrid[:] = Z0C1
                    ZINT_sgrid = np.round(Z0C_sgrid/250)
                    time_RAPscan = time_start
                    ffd_sgrid = np.copy(rlons_2d)
                    ffd_sgrid[:] = FFD1
                    storm_relative_dir = ffd_sgrid
            else:
                print('not getting new RAP data') 
                
        

        
        #Get ZDR 1km above the freezing level
        #ZDR_1km_g = np.copy(ZDRall)
        REF_1km_g = np.copy(REFall)
        KDP_1km_g = np.copy(KDPall)
        CC_1km_g = np.copy(CCall)
        for i in range(REFall.shape[0]):
#                 ZDR_1km_g1 = ZDR_1km_g[i,:,:]
#                 ZDR_1km_g1[i!=(ZINT_sgrid+4)] = 0
#                 ZDR_1km_g[i,:,:] = ZDR_1km_g1

            REF_1km_g1 = REF_1km_g[i,:,:]
            REF_1km_g1[i!=(ZINT_sgrid+4)] = 0
            REF_1km_g[i,:,:] = REF_1km_g1

            KDP_1km_g1 = KDP_1km_g[i,:,:]
            KDP_1km_g1[i!=(ZINT_sgrid+4)] = 0
            KDP_1km_g[i,:,:] = KDP_1km_g1

            CC_1km_g1 = CC_1km_g[i,:,:]
            CC_1km_g1[i!=(ZINT_sgrid+4)] = 0
            CC_1km_g[i,:,:] = CC_1km_g1

#             ZDR_1km = np.max(ZDR_1km_g, axis=0)
        REFrmasked = np.max(REF_1km_g, axis=0)
        KDPrmasked = np.max(KDP_1km_g, axis=0)
        CC_c = np.max(CC_1km_g, axis=0)

        #Get TRENDSS detections from the 1km ARL data
        #Calculating TRENDSS ZDR anomalies
        NORM_ZDR = np.copy(ZDRmasked1)
        NORM_ZDR = ma.masked_where(REF < 20, NORM_ZDR)
        refl_bins = np.arange(20, 60, 5)
        for refl_b in refl_bins:
            #print(refl_b)
            #print(REFW[0][(refl_b < REFW[0]) & (REFW[0] < (refl_b+5))])
            if np.shape(REF[(refl_b < REF) & (REF < refl_b + 5)])[0] > 20:
                #print(np.shape(REFW[i][(refl_b < REFW[i]) & (REFW[i] < refl_b + 5)]))
                #print(np.min(REFW[0][(refl_b < REFW[0]) & (REFW[0] < refl_b + 5)]))
                #print(np.nanmean(ZDRW_C[i][(refl_b < REFW[i]) & (REFW[i] < refl_b + 5)]))
                bin_arr = np.copy(ZDRmasked1[(refl_b < REF) & (REF < refl_b + 5)])
                #print(bin_arr)
                mu = np.nanmean(bin_arr)
                sigma = np.nanstd(bin_arr)
                #print(mu, sigma)
                points_c = ZDRmasked1[(refl_b < REF) & (REF < refl_b + 5)]
                points_n1 = points_c - mu
                points_n = points_n1 / sigma


                NORM_ZDR[(refl_b < REF) & (REF < refl_b + 5)] = np.asarray(points_n)

            else:
                #print('not enough points')
                mu = 10 ** ((-2.6857*(10**-4)*((refl_b+2.5)**2)) + (0.04892 * (refl_b + 2.5)) - 1.4287)
                sigma = 0.5
                #print(mu, sigma)
                points_c = ZDRmasked1[(refl_b < REF) & (REF < refl_b + 5)]
                points_n1 = points_c - mu
                points_n = points_n1 / sigma
                NORM_ZDR[(refl_b < REF) & (REF < refl_b + 5)] = np.asarray(points_n)


        #Calling gradient_section; Determining gradient direction and masking some Zhh and Zdr grid fields
        [grad_mag,grad_ffd,ZDRmasked,ZDRallmasked,ZDRrmasked] = grad_mask(Zint,REFmasked,REF,storm_relative_dir,NORM_ZDR,ZDRrmasked1,CC,CCall)
        
        #Uncomment for non-trendds ZDR arcs
#         [grad_mag,grad_ffd,ZDRmasked,ZDRmasked1,ZDRrmasked] = grad_mask(Zint,REFmasked,REF,storm_relative_dir,ZDRmasked1,ZDRrmasked1,CC,CCall)
        
        #Get ZDR 1km above the freezing level
        ZDR_1km_g = np.copy(ZDRallmasked)
        for i in range(ZDRallmasked.shape[0]):
            ZDR_1km_g1 = ZDR_1km_g[i,:,:]
            ZDR_1km_g1[i!=(ZINT_sgrid+4)] = 0
            ZDR_1km_g[i,:,:] = ZDR_1km_g1

        ZDRrmasked = np.max(ZDR_1km_g, axis=0)
        
        if np.max(VEL) > 0:
            #Calculate rotation from the velocity field
            print(VEL)
#             print(REFall)
#             print(rlons_2d)
#             print(rlats_2d)
            [az_masked, shear_maxes1, shear_maxes15, shear_maxes2, shear_maxes25, shear_lats1, shear_lats15, shear_lats2, shear_lats25, shear_lons1, shear_lons15, shear_lons2, shear_lons25, azim_save] = get_rotation(np.asarray(VEL), REFall, rlons_2d, rlats_2d, cenlon, cenlat, bin_size=7)
            
        else:
            az_masked = []
            shear_maxes1 = []
            shear_maxes15 = []
            shear_maxes2 = []
            shear_maxes25 = []
            shear_lats1 = []
            shear_lats15 = []
            shear_lats2 = []
            shear_lats25 = []
            shear_lons1 = []
            shear_lons15 = []
            shear_lons2 = []
            shear_lons25 = []


        #Let's create the ZDR column depth field as in Snyder et al. (2015)
        ZDR_count = np.copy(ZDRallmasked)
        ZDR_count[ZDR_count > 1.0] = 1
        ZDR_count[ZDR_count < 1.0] = 0

#         ZDR_sum_stuff = np.zeros((ZDR_count.shape[1], ZDR_count.shape[2]))
#         ZDR_top = np.copy(ZDR_count[(Zint-4):,:,:])
#         for i in range(ZDR_top.shape[0]):
#             ZDR_new_sum = ZDR_sum_stuff + ZDR_top[i,:,:]
#             ZDR_same = np.where(ZDR_new_sum-ZDR_sum_stuff==0)
#             ZDR_top[i:,ZDR_same[0],ZDR_same[1]] = 0
#             ZDR_sum_stuff = ZDR_new_sum

        ZDR_olaf = np.copy(ZDR_count)
        #Starting from the bottm up, set all count values below the freezing level to 0
        for i in range(ZDR_olaf.shape[0]):
            #print(i)
            ZDR_pancake = ZDR_olaf[i,:,:]
            ZDR_pancake[ZINT_sgrid>i] = 0.0
            #print(np.max(ZDR_pancake))
            ZDR_olaf[i,:,:] = ZDR_pancake

        ZDR_sum_stuff = np.zeros((ZDR_count.shape[1], ZDR_count.shape[2]))
        ZDR_top_new = np.copy(ZDR_olaf)
        for i in range(ZDR_top_new.shape[0]):
            ZDR_levthing = np.zeros((ZDR_count.shape[1], ZDR_count.shape[2]))
            ZDR_new_sum1 = ZDR_sum_stuff + ZDR_top_new[i,:,:]
            ZDR_levthing[ZINT_sgrid>i] = 1
            #print(np.max(ZDR_levthing))
            ZDR_same_new = np.where(((ZDR_new_sum1-ZDR_sum_stuff+ZDR_levthing))==0)
            ZDR_top_new[i:,ZDR_same_new[0],ZDR_same_new[1]] = 0
            ZDR_sum_stuff = ZDR_new_sum1
        #print(np.nanmax(ZDR_sum_new))

        #Let's create a field for inferred hail
        REF_Hail = np.copy(REFmasked)
        REF_Hail1 = ma.masked_where(ZDRmasked1 > 1.0, REF_Hail)
        REF_Hail2 = ma.masked_where(CC > 1.0, REF_Hail1)
        REF_Hail2 = ma.filled(REF_Hail2, fill_value = 1)

        #Let's set up the map projection!
        crs = ccrs.LambertConformal(central_longitude=-100.0, central_latitude=45.0)

        #Set up our array of latitude and longitude values and transform our data to the desired projection.
        tlatlons = crs.transform_points(ccrs.LambertConformal(central_longitude=265, central_latitude=25, standard_parallels=(25.,25.)),rlons[0,:,:],rlats[0,:,:])
        tlons = tlatlons[:,:,0]
        tlats = tlatlons[:,:,1]

        #Limit the extent of the map area, must convert to proper coords.
        LL = (st_lon-1.0,st_lat-1.0,ccrs.PlateCarree())
        UR = (st_lon+1.0,st_lat+1.0,ccrs.PlateCarree())
        print(LL)

        #Get data to plot state and province boundaries
        states_provinces = cfeature.NaturalEarthFeature(
                category='cultural',
                name='admin_1_states_provinces_lakes',
                scale='50m',
                facecolor='none')
        #Make sure these shapefiles are in the same directory as the script
        fname = 'cb_2016_us_county_20m/cb_2016_us_county_20m.shp'
        fname2 = 'cb_2016_us_state_20m/cb_2016_us_state_20m.shp'
        counties = ShapelyFeature(Reader(fname).geometries(),ccrs.PlateCarree(), facecolor = 'none', edgecolor = 'black')
        states = ShapelyFeature(Reader(fname2).geometries(),ccrs.PlateCarree(), facecolor = 'none', edgecolor = 'black')

        #Create a figure and plot up the initial data and contours for the algorithm
        fig=plt.figure(n,figsize=(30.,25.))
        ax = plt.subplot(111,projection=ccrs.PlateCarree())
        ax.coastlines('50m',edgecolor='black',linewidth=0.75)
        ax.add_feature(counties, edgecolor = 'black', linewidth = 0.5)
        ax.add_feature(states, edgecolor = 'black', linewidth = 1.5)
        ax.set_extent([LL[0],UR[0],LL[1],UR[1]])
        REFlevels = np.arange(20,73,2)
        depth_levels= np.arange(0.01,23,1)

        #Options for Z backgrounds/contours
        #refp = ax.pcolormesh(ungrid_lons, ungrid_lats, ref_c, cmap=plt.cm.gist_ncar, vmin = 10, vmax = 73)
        #refp = ax.pcolormesh(ungrid_lons, ungrid_lats, ref_ungridded_base, cmap='HomeyerRainbow', vmin = 10, vmax = 73)
        refp = ax.pcolormesh(rlons_2d, rlats_2d, REFmasked, cmap=pyart.graph.cmweather.cm_colorblind.HomeyerRainbow, vmin = 10, vmax = 73)
        refp2 = ax.contour(rlons_2d, rlats_2d, REFmasked, [40], colors='grey', linewidths=5, zorder=1)
        #refp3 = ax.contour(rlons_2d, rlats_2d, REFmasked, [45], color='r')
        #plt.contourf(rlons_2d, rlats_2d, ZDR_sum_stuff, depth_levels, cmap=plt.cm.viridis)

        ncfile = Dataset('SPORK_NC/SPORK_ONE'+str(year)+str(month)+str(day)+str(hour)+str(station)+'.nc',mode='w',format='NETCDF4_CLASSIC') 

        #Save as a netCFD file
        lat_dim = ncfile.createDimension('lat', REFall.shape[1])     # latitude axis
        lon_dim = ncfile.createDimension('lon', REFall.shape[2])    # longitude axis
        level_dim = ncfile.createDimension('level', REFall.shape[0])    # longitude axis
        time_dim = ncfile.createDimension('time', None) # unlimited axis (can be appended to).

        # Define a 4D variable to hold the data
        temp = ncfile.createVariable('REFL',np.float64,('level','lat','lon')) # note: unlimited dimension is leftmost
        temp.units = 'dBZ' # degrees Kelvin
        temp.standard_name = 'REFL_10CM' # this is a CF standard name

        # Write the data.  This writes the whole 3D netCDF variable all at once.
        temp[:,:,:] = REFall # Appends data along unlimited dimension

        # Define a 4D variable to hold the data
        temp12 = ncfile.createVariable('VEL',np.float64,('level','lat','lon')) # note: unlimited dimension is leftmost
        temp12.units = 'm s-1' # degrees Kelvin
        temp12.standard_name = 'VEL' # this is a CF standard name

        # Write the data.  This writes the whole 3D netCDF variable all at once.
        temp12[:,:,:] = np.asarray(VEL) # Appends data along unlimited dimension


        # Define a 4D variable to hold the data
        temp2 = ncfile.createVariable('ZDR',np.float64,('level','lat','lon')) # note: unlimited dimension is leftmost
        temp2.units = 'dB' # degrees Kelvin
        temp2.standard_name ='ZDR' # this is a CF standard name

        # Write the data.  This writes the whole 3D netCDF variable all at once.
        temp2[:,:,:] = ZDRall # Appends data along unlimited dimension

        # Define a 4D variable to hold the data
        temp6 = ncfile.createVariable('KDP',np.float64,('level','lat','lon')) # note: unlimited dimension is leftmost
        temp6.units = 'deg km-1' # degrees Kelvin
        temp6.standard_name ='KDP' # this is a CF standard name

        # Write the data.  This writes the whole 3D netCDF variable all at once.
        temp6[:,:,:] = KDPall # Appends data along unlimited dimension

        # Define a 4D variable to hold the data
        temp7 = ncfile.createVariable('CC',np.float64,('level','lat','lon')) # note: unlimited dimension is leftmost
        temp7.units = 'dB' # degrees Kelvin
        temp7.standard_name ='CC' # this is a CF standard name

        # Write the data.  This writes the whole 3D netCDF variable all at once.
        temp7[:,:,:] = CCall # Appends data along unlimited dimension

        # Define a 4D variable to hold the data
        temp10 = ncfile.createVariable('ROT',np.float64,('level','lat','lon')) # note: unlimited dimension is leftmost
        temp10.units = 's-1' # degrees Kelvin
        temp10.standard_name ='ROT' # this is a CF standard name

        # Write the data.  This writes the whole 3D netCDF variable all at once.
        temp10[:,:,:] = az_masked / 493 # Appends data along unlimited dimension

        # Define a 3D variable to hold the data for ZDR depth
        temp8 = ncfile.createVariable('ZDRD',np.float64,('lat','lon')) # note: unlimited dimension is leftmost
        temp8.standard_name = 'ZDRD' # this is a CF standard name

        # Write the data.  This writes the whole 3D netCDF variable all at once.
        temp8[:,:] = ZDR_sum_stuff # Appends data along unlimited dimension

        # Define a 3D variable to hold the data for ZDR depth
        temp9 = ncfile.createVariable('NZDR',np.float64,('lat','lon')) # note: unlimited dimension is leftmost
        temp9.standard_name = 'NZDR' # this is a CF standard name

        # Write the data.  This writes the whole 3D netCDF variable all at once.
        temp9[:,:] = NORM_ZDR # Appends data along unlimited dimension

        # Define a 3D variable to hold the data for ZDR depth
        temp3 = ncfile.createVariable('Lons',np.float64,('lat','lon')) # note: unlimited dimension is leftmost
        temp3.standard_name = 'Lons' # this is a CF standard name


        # Write the data.  This writes the whole 3D netCDF variable all at once.
        temp3[:,:] = rlons_2d # Appends data along unlimited dimension
        
        # Define a 3D variable to hold the data for ZDR depth
        temp11 = ncfile.createVariable('azim',np.float64,('lat','lon')) # note: unlimited dimension is leftmost
        temp11.standard_name = 'azim' # this is a CF standard name

        # Write the data.  This writes the whole 3D netCDF variable all at once.
        temp11[:,:] = azim_save # Appends data along unlimited dimension

        # Define a 3D variable to hold the data for ZDR depth
        temp4 = ncfile.createVariable('Lats',np.float64,('lat','lon')) # note: unlimited dimension is leftmost
        temp4.standard_name = 'Lats' # this is a CF standard name

        # Write the data.  This writes the whole 3D netCDF variable all at once.
        temp4[:,:] = rlats_2d # Appends data along unlimited dimension

        # Define a 3D variable to hold the data for ZDR depth
        temp5 = ncfile.createVariable('Times',np.float64,('time')) # note: unlimited dimension is leftmost
        temp5.standard_name = 'Times' # this is a CF standard name


        # Write the data.  This writes the whole 3D netCDF variable all at once.
        temp5[:] = dt.timestamp() # Appends data along unlimited dimension

        print(time_start)

        break

    return REFall, ZDRall, KDPall, CCall, NORM_ZDR, az_masked, rlons_2d, rlats_2d
