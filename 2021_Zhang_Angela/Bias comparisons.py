#!/usr/bin/env python

'''
Plot gyro and accelrometer bias for the SITL logs
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

args = parser.parse_args()
param = args.param

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
f = open('EqFoutput_' + args.log + ' EKparam ' + param + '.pckl', 'rb')
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


#Compare relative N,E,D positions

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

R0 = rotmat_xyz2NED(output['SIM.lat'][0],output['SIM.long'][0])

earthrad = 6.3781*1.0e6 #mean radius is 6371km, but here follows convention used in mp_util.py from Mavproxy module
p = lla2xyz(output['SIM.lat'],output['SIM.long'],np.array(output['SIM.alt']),earthrad)
p_rel = R0 @ p

#using AP default SIM2 PN PE and PD as comparison
sim2_N = np.array(output['SIM2.Xi_p'])[:,0,0]
sim2_N = sim2_N - sim2_N[0]
sim2_E = np.array(output['SIM2.Xi_p'])[:,1,0]
sim2_E = sim2_E - sim2_E[0]
sim2_D = np.array(output['SIM2.Xi_p'])[:,2,0]
sim2_D = sim2_D - sim2_D[0]
p_rel1 = np.vstack((sim2_N,sim2_E,sim2_D))##########

pos_rel = R0 @ (p.T - p[:,0]).T

def interp_func(x,y,newx):
    f = interp1d(x,y)
    newy = f(newx)
    return newy


#Plot gyro bias
fig4, ax4 = plt.subplots(3)
ax4[0].plot(output['EK3.time'], np.array(output['EK3.biasgx'])*(pi/180),c = 'mediumblue')
ax4[0].plot(output['EqF.time'], output['EqF.biasgx'],c = 'orangered', ls='--')
ax4[0].axhline(y=EqFbias[0][0],xmin=0,xmax= output['EqF.time'][-1], c = 'g', ls = '-.', linewidth=1)
fig4.suptitle('Gyroscope Bias Estimate')
ax4[0].set_ylabel('X axis')
ax4[1].plot(output['EK3.time'], np.array(output['EK3.biasgy'])*(pi/180),c = 'mediumblue')
ax4[1].plot(output['EqF.time'], output['EqF.biasgy'],c = 'orangered', ls='--')
ax4[1].axhline(y=EqFbias[1][0],xmin=0,xmax= output['EqF.time'][-1],c='g',ls = '-.',linewidth=1)
ax4[1].set_ylabel('Y axis')
ax4[2].plot(output['EK3.time'], np.array(output['EK3.biasgz'])*(pi/180),c = 'mediumblue',label='EKF3')
ax4[2].plot(output['EqF.time'], output['EqF.biasgz'],c = 'orangered', ls='--',label='EqF')
ax4[2].axhline(y=EqFbias[2][0],xmin=0,xmax= output['EqF.time'][-1],c='g',ls = '-.',linewidth=1,label='True Gyroscope Bias')
ax4[2].legend(loc='best', prop={'size': 8})
ax4[2].set_xlabel('Time (s)')
ax4[2].set_ylabel('Z axis Estimate (rad$s^{-1}$)')
plt.setp(ax4, xlim=[0,output['EK3.time'][-1]])
figure = plt.gcf()  # get current figure
figure.set_size_inches(8, 5) # set figure's size manually to your full screen
fig4.savefig('Graphs for simulation set 2/'+ args.log+'/Gyro Bias ' + args.log + '.pdf', bbox_inches='tight')

#Plot accel bias
fig5, ax5 = plt.subplots(3)
ax5[0].plot(output['XKF.time'], output['EK3.biasax'],c = 'mediumblue')
ax5[0].plot(output['EqF.time'], output['EqF.biasax'],c = 'orangered', ls='--')
fig5.suptitle('Accelerometer Bias Estimate')
ax5[0].axhline(y=EqFbias[3][0],xmin=0,xmax= output['EqF.time'][-1],c = 'g', ls = '-.', linewidth=1)
ax5[0].set_ylabel('X axis')
ax5[1].plot(output['XKF.time'], output['EK3.biasay'],c = 'mediumblue')
ax5[1].plot(output['EqF.time'], output['EqF.biasay'],c = 'orangered', ls='--')
ax5[1].axhline(y=EqFbias[4][0],xmin=0,xmax= output['EqF.time'][-1],c = 'g', ls = '-.', linewidth=1)
ax5[1].set_ylabel('Y axis')
ax5[2].plot(output['XKF.time'], output['EK3.biasaz'],c = 'mediumblue',label='EKF3')
ax5[2].plot(output['EqF.time'], output['EqF.biasaz'],c = 'orangered', ls='--',label='EqF')
ax5[2].axhline(y=EqFbias[5][0],xmin=0,xmax= output['EqF.time'][-1],c = 'g', ls = '-.', linewidth=1,label='True Accelerometer Bias')
ax5[2].legend(loc='best', prop={'size': 8})
ax5[2].set_xlabel('Time(s)')
ax5[2].set_ylabel('Z axis Estimate ($ms^{-2}$)')
plt.setp(ax5, xlim=[0,output['EK3.time'][-1]])
figure = plt.gcf()  # get current figure
figure.set_size_inches(8, 5) # set figure's size manually to your full screen
fig5.savefig('Graphs for simulation set 2/'+ args.log+'/Accel Bias ' + args.log + '.pdf', bbox_inches='tight')


j = np.searchsorted(output['EqF.GPStime'],output['SIM.time'][0])
k = np.searchsorted(output['EqF.GPStime'],output['SIM.time'][-1],side ='right')
Xi_p = np.empty([3,len(output['EqF.GPStime'][j:k])])
Xi_v = np.empty([3,len(output['EqF.GPStime'][j:k])])

for i in range(3):    
    #interpolate SITL position to match EqF time
    Xi_p[i,:] = interp_func(np.array(output['SIM.time']),pos_rel[i,:], np.array(output['EqF.GPStime'][j:k]))         
    #interpolate SITL velocity to match EqF time
    Xi_v[i,:] = interp_func(np.array(output['SIM2.time']),np.array(output['SIM2.Xi_v'])[:,i,0], np.array(output['EqF.GPStime'][j:k]))         

##For thesis presentation
#plot only the position
plt.figure()
plt.plot(output['EqF.GPStime'][j:k], Xi_p[0,:],label='SITL position')
plt.legend(loc='best')
plt.title('Simulation true position')
plt.xlabel('Time (s)')
plt.ylabel('North (m)')
plt.show()
figure = plt.gcf()  # get current figure
figure.set_size_inches(8, 3) # set figure's size manually to your full screen (32x18)
plt.savefig('Graphs for simulation set 2/'+ args.log+'/True North pos' + args.log + '.pdf', bbox_inches='tight')

#plot only the velocity
plt.figure()
plt.plot(output['EqF.GPStime'][j:k], Xi_v[0,:],label='SITL velocity')
plt.legend(loc='best')
plt.title('Simulation true velocity')
plt.xlabel('Time (s)')
plt.ylabel('North (m$s^{-1}$)')
plt.show()
figure = plt.gcf()  # get current figure
figure.set_size_inches(8, 3) # set figure's size manually to your full screen
plt.savefig('Graphs for simulation set 2/'+ args.log+'/True North vel' + args.log + '.pdf', bbox_inches='tight')
   