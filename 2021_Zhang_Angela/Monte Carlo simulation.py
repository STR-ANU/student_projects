#!/usr/bin/env python

'''
Plot the Monte Carlo simulations for NEES and NIS
'''
from __future__ import print_function
from builtins import range

import os

from MAVProxy.modules.lib import mp_util
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("log", metavar="LOG")
parser.add_argument("--debug", action='store_true')
parser.add_argument('--param', type= str, default = 'true', help='type true to set param same as EKF')
parser.add_argument('--addnoise', type= str, default = 'true', help='type true to use file with noise added')
parser.add_argument("--simnum", type=int, default=10, help='Number of simulations for Monte Carlo, default is 10')

args = parser.parse_args()
param = args.param
addnoise = args.addnoise
trial_num = args.simnum

from pymavlink import mavutil
from pymavlink.rotmat import Vector3, Matrix3
from pymavlink.mavextra import expected_earth_field_lat_lon
import math
from math import degrees

GRAVITY_MSS = 9.80665

import numpy as np
import scipy
from scipy.linalg import expm
from scipy.linalg import logm
from math import pi
import random
import utm

from scipy.interpolate import interp1d

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from statistics import mean

import subprocess
from scipy.stats.distributions import chi2

#Run 'trial_num' times for Monte Carlo simulation
#Comment out line 53-54 if want to use the pre-uploaded pickle files for Monte Carlo analysis
for i in range(trial_num):
    subprocess.run(['python.exe', 'Monte_Carlo_EqF.py', str(args.log), str(i), '--param', 'true'])

def skew_to_vec (skew):
    return np.array([[skew[2][1]],[skew[0][2]],[skew[1][0]]])

def gps_diff(pos1, pos2):
    '''return the distance from pos1 to pos2, both GPS position
       pos1 and pos2 are Vector3
    '''
    dist = mp_util.gps_distance(pos1.x, pos1.y, pos2.x, pos2.y) #return distance between two points in meters, coordinates are in degrees
    bearing_rad = math.radians(mp_util.gps_bearing(pos1.x, pos1.y, pos2.x, pos2.y)) #return bearing between two points in degrees, in range 0-360
    dx = math.cos(bearing_rad) * dist
    dy = math.sin(bearing_rad) * dist
    # GPS is +ve up, we want +ve down
    dz = pos1.z - pos2.z
    return Vector3(dx, dy, dz)

###Load SITL, EqF and EKF pickle data for comparison
import pickle

if param == 'true':
    print("Using EKF3 parameter")
elif param =='false':
    print("Using default parameter")
else:
    print('Error! Not sure what\'s the param setting')

f = open('SITLoutput_' + args.log + '.pckl', 'rb')
sitl = pickle.load(f)
f.close()

if addnoise == 'true':
        print("Using manually added noise EqF file")
else:
        assert 1==0, "error, did not input add noise command"
        print("Error! Check code")


allNEES = []
allgpsNIS = []
allaltNIS = []
for i in range(trial_num):
    f = open('Monte Carlo EqFoutput_' + args.log+ str(i)+ ' EKparam ' + param + '.pckl', 'rb')
    EqF = pickle.load(f)
    f.close()

    output = {**sitl, **EqF}
        
    filename = args.log
    
    print("Using log number %s" %i)
    
    if "nobias" in args.log:
        EqFbias = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
        print("Using 0 for true bias")
    else:
        EqFbias = np.array([[-0.01],[-0.02],[-0.03],[0.1],[0.2],[0.2]])
        print("Setting true bias value")
    
    def interp_func(x,y,newx):
        f = interp1d(x,y)
        newy = f(newx)
        return newy
    
    def rotmat_xyz2NED(lat, long):
        R1 = np.array([-math.sin(math.radians(lat)) * math.cos(math.radians(long)),
                        -math.sin(math.radians(lat)) * math.sin(math.radians(long)),
                        math.cos(math.radians(lat))
                        ])
        R2 = np.array([-math.sin(math.radians(long)),
                        math.cos(math.radians(long)),
                        0
                        ])
        R3 = np.array([-math.cos(math.radians(lat)) * math.cos(math.radians(long)),
                        -math.cos(math.radians(lat)) * math.sin(math.radians(long)),
                        -math.sin(math.radians(lat))
                        ])
        
        R_0 = np.array([R1,R2,R3])
        return R_0
    
    def lla2xyz(lat,long,alt,earthradius): #converts lat,long,alt coord to x,y,z coord
        p = (alt+ earthradius) * np.array([
                np.cos(np.radians(lat)) * np.cos(np.radians(long)),
                np.cos(np.radians(lat)) * np.sin(np.radians(long)),
                np.sin(np.radians(lat))
                ])
        return p
    
    #Plot NEES
    #SITL relative position
    R0 = rotmat_xyz2NED(output['SIM.lat'][0],output['SIM.long'][0])
    earthrad = 6.3781*1.0e6 #mean radius is 6371km, but here follows convention used in mp_util.py from Mavproxy module
    p = lla2xyz(output['SIM.lat'],output['SIM.long'],np.array(output['SIM.alt']),earthrad)
    pos_rel = R0 @ (p.T - p[:,0]).T
    
    SITL_time = np.array(output['SIM.time'])
    SIM_lat = np.array(output['SIM.lat'])
    SIM_long = np.array(output['SIM.long'])
    SIM_alt = np.array(output['SIM.alt'])
    SIM_relativepos_EqF = np.empty([3,len(SITL_time)])
    for i in range(len(SITL_time)):
        SIM_position = Vector3(SIM_lat[i], SIM_long[i], SIM_alt[i])
        x_p = gps_diff(output['EqF.origin'][0], SIM_position)
        SIM_relativepos_EqF[:,i] = [x_p.x, x_p.y, x_p.z]
    
    #First compute EqF NEES
    j = np.searchsorted(output['EqF.GPStime'],output['SIM.time'][0])
    k = np.searchsorted(output['EqF.GPStime'],output['SIM.time'][-1],side ='right')
    Xi_p = np.empty([3,len(output['EqF.GPStime'][j:k])])
    Xi_v = np.empty([3,len(output['EqF.GPStime'][j:k])])
    
    for i in range(3):
        #interpolate SITL position obtained using the gps diff functino to match EqF time
        Xi_p[i,:] = interp_func(np.array(output['SIM.time']),SIM_relativepos_EqF[i,:], np.array(output['EqF.GPStime'][j:k]))
        #interpolate SITL velocity to match EqF time
        Xi_v[i,:] = interp_func(np.array(output['SIM2.time']),np.array(output['SIM2.Xi_v'])[:,i,0], np.array(output['EqF.GPStime'][j:k]))         
    p_inv = np.array(output['EqF.p_inv'][j:k])
    v_inv = np.array(output['EqF.v_inv'][j:k])
    
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp
    #Interpolate SITL rotation to match EqF time
    def slerp_interp(x,y,newx):
        rotation = R.from_matrix(np.array(y))
        slerp = Slerp(np.array(x), rotation)
        interp_rot = slerp(newx)
        return interp_rot.as_matrix()
    Xi_R = slerp_interp(output['SIM.time'],output['SIM.Xi_R'], output['EqF.GPStime'][j:k])
    R_inv = np.array(output['EqF.R_inv'][j:k])
    
    #Apply right action for rotation and apply error map
    R_action = Xi_R @ R_inv
    epsilon_R = [logm(i) for i in R_action]
    epsilon_R = np.array([skew_to_vec(i) for i in epsilon_R]).T.reshape(3,-1)
    #Apply right action to position and velocity, error map they are kept the same
    epsilon_p = Xi_p + (Xi_R @ p_inv).T.reshape(3,-1)
    epsilon_v = Xi_v + (Xi_R @ v_inv).T.reshape(3,-1)
    
    epsilon = np.vstack((epsilon_R, epsilon_p, epsilon_v))
    bias_error = EqFbias - (np.array(output['EqF.bhat'][j:k])).T.reshape(6,-1)
    error = np.vstack((epsilon, bias_error))
    ric = np.linalg.inv(np.array(output['EqF.Riccati'][j:k]))
    eqf_e = error.T.reshape(len(output['EqF.GPStime'][j:k]),len(error),1)
    eqf_eT = error.T.reshape(len(output['EqF.GPStime'][j:k]),1,len(error))
    EqF_NEES = (eqf_eT @ ric @ eqf_e).reshape(k-j)
    
    allNEES.append(EqF_NEES)
    allgpsNIS.append(output['EqF.gpsNIS'])
    allaltNIS.append(output['EqF.altNIS'])


avg_EqF_NEES = np.mean(np.array(allNEES),axis=0)
avg_gpsNIS = np.mean(np.array(allgpsNIS),axis=0)
avg_altNIS = np.mean(np.array(allaltNIS),axis=0)

plt.figure()
plt.plot(output['EqF.GPStime'][j:k], avg_EqF_NEES, label = 'EqF NEES (all 15 states)')
plt.title('Average NEES for %i Monte Carlo runs' %(trial_num))
lowerbound = chi2.ppf(0.025, df=(15*trial_num))/trial_num
upperbound = chi2.ppf(0.975, df=(15*trial_num))/trial_num
plt.axhline(y=15, color='b', linestyle='-.', label = 'E[dim(states)]')
plt.axhline(lowerbound, color='r', linestyle='-.', label = 'EqF bounds')
plt.axhline(upperbound, color='r', linestyle='-.')
plt.legend(loc='best')
plt.yscale("log")
plt.xlabel('Time(s)')
figure = plt.gcf()  # get current figure
figure.set_size_inches(8,5) # set figure's size manually to your full screen (32x18)
plt.savefig('Monte Carlo NEES ' + args.log + '.png', bbox_inches='tight') # bbox_inches removes extra white spaces


##Compute EKF3 NIS and plot together with EqF NIS 
#Plot GPS NIS
plt.figure()
plt.plot(output['EqF.GPStime'],avg_gpsNIS,label = 'EqF GPS NIS')
plt.title('Horizontal position NIS for %i Monte Carlo runs' %trial_num)
plt.axhline(y=2, color='b', linestyle='-.', label = 'E[GPS NIS]')
lowerbound = chi2.ppf(0.025, df=(2*trial_num))/trial_num
upperbound = chi2.ppf(0.975, df=(2*trial_num))/trial_num
plt.axhline(lowerbound, color='r', linestyle='-.', label = 'Lower and upper Bounds')
plt.axhline(upperbound, color='r', linestyle='-.')
plt.legend(loc='best')
plt.yscale("log")
plt.xlabel('Time(s)')
plt.ylabel('GPS NIS')
figure = plt.gcf()  # get current figure
figure.set_size_inches(8, 5) # set figure's size manually to your full screen
plt.savefig('Monte Carlo GPS NIS ' + args.log + '.png', bbox_inches='tight') # bbox_inches removes extra white spaces

#Plot Altitude NIS
plt.figure()
plt.plot(output['baro.time'],avg_altNIS,label = 'EqF alt NIS')
plt.title('Vertical position NIS for %i Monte Carlo runs' %trial_num)
plt.axhline(y=1, color='b', linestyle='-.', label = 'E[Altitude NIS]')
lowerbound = chi2.ppf(0.025, df=(1*trial_num))/trial_num
upperbound = chi2.ppf(0.975, df=(1*trial_num))/trial_num
plt.axhline(lowerbound, color='r', linestyle='-.', label = 'Lower and upper bounds')
plt.axhline(upperbound, color='r', linestyle='-.')
plt.legend(loc='best')
plt.yscale("log")
plt.xlabel('Time(s)')
plt.ylabel('Altitude NIS')
figure = plt.gcf()  # get current figure
figure.set_size_inches(8,5) # set figure's size manually to your full screen
plt.savefig('Monte Carlo Alt NIS ' + args.log + '.png', bbox_inches='tight') # bbox_inches removes extra white spaces