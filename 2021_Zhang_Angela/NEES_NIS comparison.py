#!/usr/bin/env python

'''
Plot the NEES and NIS of logs
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
parser.add_argument('--addnoise', type= str, default = 'false', help='type true to use file with noise manually added')

args = parser.parse_args()
param = args.param
addnoise = args.addnoise

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
#plt.close("all")

from statistics import mean

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
    f = open('EqFoutput_' + args.log + '_addnoise_EKparam ' + param + '.pckl', 'rb')
    print("Using manually added noise EqF file")
else:
    f = open('EqFoutput_' + args.log + ' EKparam ' + param + '.pckl', 'rb')
    print("Using no extra noise EqF file")
EqF = pickle.load(f)
f.close()
f = open('EKFoutput_' + args.log + '.pckl', 'rb')
ekf = pickle.load(f)
f.close()

output = {**sitl, **EqF, **ekf}
filename = args.log

if "nobias" in args.log:
    EqFbias = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
    EKFinternal_bias = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
    print("Using 0 for true bias")
else:
    EqFbias = np.array([[-0.01],[-0.02],[-0.03],[0.1],[0.2],[0.2]])
    dtavg = output['EK3.dtaverage'][0]
    EKFinternal_bias = dtavg * np.array([[-0.01],[-0.02],[-0.03],[0.1],[0.2],[0.2]])
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


#Compute EKF NEES
#Use SVAA.time as base time
m = np.searchsorted(output['SVAA.time'],output['SIM.time'][0])
n = np.searchsorted(output['SVAA.time'],output['SIM.time'][-1],side ='right')

#Interpoalte SITL position and velocity to match SVAA.time
SITL_time = np.array(output['SIM.time'])
SIM_lat = np.array(output['SIM.lat'])
SIM_long = np.array(output['SIM.long'])
SIM_alt = np.array(output['SIM.alt'])
SIM_relativepos_EKF = np.empty([3,len(SITL_time)])
for i in range(len(SITL_time)):
    EKF_origin = Vector3(output['EK3org.lat'][0],output['EK3org.long'][0],output['EK3org.alt'][0])
    SIM_position = Vector3(SIM_lat[i], SIM_long[i], SIM_alt[i])
    x_p = gps_diff(EKF_origin, SIM_position)
    SIM_relativepos_EKF[:,i] = [x_p.x, x_p.y, x_p.z]
    
#Because number of SVAA time stamp is large, reduce by cutting in timesteps
step = 10
Xi2_p = np.empty([3,len(output['SVAA.time'][m:n:step])])
Xi2_v = np.empty([3,len(output['SVAA.time'][m:n:step])])
for i in range(3):     
    #interpolate SITL position obtained using the gps diff functino to match EqF time
    Xi2_p[i,:] = interp_func(np.array(output['SIM.time']),SIM_relativepos_EKF[i,:], np.array(output['SVAA.time'][m:n:step]))
    #interpolate SITL velocity
    Xi2_v[i,:] = interp_func(np.array(output['SIM2.time']),np.array(output['SIM2.Xi_v'])[:,i,0], np.array(output['SVAA.time'][m:n:step]))         

#Interpolate SITL quaternion to match SVAA.time
def slerp_interp_quat(x,y,newx):
    y = np.roll(np.array(y),-1, axis =1) #rotation object store real (scalar) components last
    quaternions = R.from_quat(np.array(y))
    slerp = Slerp(np.array(x), quaternions)
    interp_quat = slerp(newx)
    return np.roll(interp_quat.as_quat(),1, axis = 1)
SIM_quat = slerp_interp_quat(output['SIM.time'],output['SIM.quaternion'], output['SVAA.time'][m:n:step])

#EKF NEES
EKFstate_0_14 = np.array(output['SVAA.state'][m:n:step]) #Cut EKF states to match SVAA time
EKFstate_15 = np.array(output['SVAB.state'][m:n:step])
length = len(range(m,n,step)) #work out the length of new time

#Compute the quaternion difference
ekf_quat = EKFstate_0_14[:,0:4]
from pyquaternion import Quaternion 
interp_time = output['SVAA.time'][m:n:step]
quaternion_diff = np.empty([len(interp_time),4])
for i in range(len(interp_time)):
    #Find the two quaternions to interpolate in between
    index = np.searchsorted(output['SIM.time'],interp_time[i])
    q1 = Quaternion(np.array(output['SIM.quaternion'][index-1]))
    q2 = Quaternion(np.array(output['SIM.quaternion'][index]))
    q1 = q1.normalised #make sure it's normalised
    q2 = q2.normalised
    
    #Interpolate between unit quaternions by t_fraciton = [0,1], 0 being q1 and 1 being q2
    t_fraction = (interp_time[i] - output['SIM.time'][index-1])/ (output['SIM.time'][index] - output['SIM.time'][index-1])
    q = Quaternion.slerp(q1, q2, t_fraction)
    true_q = q.normalised

    #Check if the ekf quaternion needs to be multiplied by -1
    ekf_q = Quaternion(ekf_quat[i,:])
    ekf_q = ekf_q.normalised
    quat_dot = true_q.elements[0]*ekf_q.elements[0] + true_q.elements[1]*ekf_q.elements[1] + \
                true_q.elements[2]*ekf_q.elements[2] + true_q.elements[3]*ekf_q.elements[3]
    if quat_dot < 0:
        ekf_q *= -1
    q_hat = ekf_q.normalised #make sure quaternion is normalised
    
    quaternion_diff[i,:] = true_q.elements - q_hat.elements

#Compute difference for vel, pos and bias
bias = np.tile(EKFinternal_bias,length)
SITL_velposbias = np.vstack((Xi2_v, Xi2_p, bias))
EKF_velposbias = np.vstack((EKFstate_0_14[:,4:].T, EKFstate_15.T))
err_velposbias = EKF_velposbias - SITL_velposbias

#Error for all 16 states and inverse the covariance
er = np.vstack((quaternion_diff.T, err_velposbias))
covariance = np.array(output['EKF.Covariances'][m:n:step])
cov_trim = np.linalg.pinv(covariance[:,:,:])
cov_trim = cov_trim[:,[*range(0,16)],:]
cov_trim = cov_trim[:,:,[*range(0,16)]]
ekf_e = er.T.reshape(length, len(er), 1)
ekf_eT = er.T.reshape(length, 1, len(er))
ekf_NEES = (ekf_eT @ cov_trim @ ekf_e).reshape(length) #EKF NEES for the first 16 states

EKFtime = np.array(output['SVAA.time'][m:n:step])
if addnoise != 'true': 
    plt.figure()
    plt.plot(output['EqF.GPStime'][j:k], EqF_NEES, '--', c = 'orangered', label = 'EqF NEES (All 15 states)')
    trim = np.logical_and(ekf_NEES >0, ekf_NEES>0)
    ekf_NEES = ekf_NEES[trim]
    EKFtime_trim = EKFtime[trim]
    plt.plot(EKFtime_trim[50:], ekf_NEES[50:], c = 'mediumblue', label = 'EKF NEES (16 states for rotation, position, velocity and bias)')
    plt.title('Normalized Estimation Error Squared (NEES)')
    plt.legend(loc='best')
    plt.yscale("log")
    plt.xlabel('Time (s)')
    plt.ylabel('NEES')
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 5) # set figure's size manually to your full screen (32x18)
    plt.savefig('Graphs for simulation set 2/'+ args.log+'/NEES noiseless ' + args.log + '.pdf', bbox_inches='tight')
elif addnoise == 'true':
    plt.figure()
    plt.plot(output['EqF.GPStime'][j:k], EqF_NEES, '--', c='orangered', label = 'EqF NEES (All 15 states)')
    plt.title('Normalized estimation error squared (NEES) (on %s log)' %filename)
    plt.axhline(y=6.26, color='g', linestyle='-.', label = 'EqF bounds') #bounds for NEES assuming chi-squared distribution for 1 simulation
    plt.axhline(y=27.49, color='g', linestyle='-.')
    plt.legend(loc='best')
    plt.yscale("log")
    plt.xlabel('Time (s)')
    plt.ylabel('NEES')
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 5) # set figure's size manually to your full screen (32x18)
    plt.savefig('Graphs for simulation set 2/'+ args.log+'/NEES with noise ' + args.log + '.pdf', bbox_inches='tight')

#Plot state estimate error squared
error_sqr = (eqf_eT @ eqf_e).reshape(k-j)
EKF_error_sqr = (ekf_eT @ ekf_e).reshape(length)

plt.figure()
plt.title('State Estimate Error Squared')
plt.plot(output['EqF.GPStime'][j:k], error_sqr,'--', c='orangered', label = 'EqF error squared')
plt.plot(EKFtime, EKF_error_sqr, c='mediumblue',label = 'EKF error squared')
plt.legend(loc='best')
plt.yscale("log")
plt.ylabel('State Error Squared')
plt.xlabel('Time (s)')
figure = plt.gcf()  # get current figure
figure.set_size_inches(8, 3) # set figure's size manually to your full screen
plt.savefig('Graphs for simulation set 2/'+ args.log+'/State Error squared ' + args.log + '.pdf', bbox_inches='tight')

##Compute EKF3 NIS and plot together with EqF NIS 
#Plot GPS NIS
if addnoise != 'true': 
    plt.figure()
    plt.plot(output['EqF.GPStime'],output['EqF.gpsNIS'],'--', c='orangered',label = 'EqF')
    plt.plot(output['INAC.time'], output['EKF.GPSnis'],c='mediumblue',label = 'EKF3')
    plt.title('Horizontal Position Normalized Innovation Squared (NIS)')
    plt.legend(loc='best')
    plt.yscale("log")
    plt.xlabel('Time (s)')
    plt.ylabel('NIS')
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 5) # set figure's size manually to your full screen
    plt.savefig('Graphs for simulation set 2/'+ args.log+'/GPS NIS noiseless ' + args.log + '.pdf', bbox_inches='tight')
elif addnoise =='true':
    plt.figure()
    plt.plot(output['EqF.GPStime'],output['EqF.gpsNIS'],'--', c='orangered',label = 'EqF')
    plt.title('Horizontal Position Normalized Innovation Squared (NIS)')
    plt.axhline(y=2, color='r', linestyle='-.', label = 'Expected NIS Value')
    plt.legend(loc='best')
    plt.yscale("log")
    plt.xlabel('Time (s)')
    plt.ylabel('NIS')
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 5) # set figure's size manually to your full screen
    plt.savefig('Graphs for simulation set 2/'+ args.log+'/GPS NIS with noise ' + args.log + '.pdf', bbox_inches='tight')