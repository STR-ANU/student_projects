#!/usr/bin/env python

'''
Plot the 3D position, position and velocity (in N,E,D components) for the real flight log
'''
from __future__ import print_function
from builtins import range

import os

from MAVProxy.modules.lib import mp_util
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("log", metavar="LOG")
parser.add_argument('--param', type= str, default = 'true', help='type false to set param as EqF default (change the values in code to what you want)')
parser.add_argument('--correct-gps', action='store_true', help='Correct GPS by half sample')
parser.add_argument('--gps-num', type=int, default=0, help='GPS selection')
parser.add_argument("--debug", action='store_true')

args = parser.parse_args()
param_setting = args.param
gps_select = args.gps_num
import subprocess
subprocess.run(['python.exe', 'Real Data EqF_processing.py', str(args.log), '--param', str(param_setting), '--correct-gps', '--gps-num', str(gps_select)])

from pymavlink import mavutil
from pymavlink.rotmat import Vector3, Matrix3
from pymavlink.mavextra import expected_earth_field_lat_lon
import math
from math import degrees

import pickle

GRAVITY_MSS = 9.80665
param = args.param

import numpy as np
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

f = open('EqFoutput_' + 'qubit-springvalley-2021-06-06-flight1.bin' + ' EKparam ' + param + '.pckl', 'rb')
EqFoutput = pickle.load(f)
f.close()

filename = args.log

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

#RTK relative position using the lla2xyz method
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

earthrad = 6.3781*1.0e6 #mean radius is 6371km, but here follows convention used in mp_util.py from Mavproxy module
RTKpostime = np.array(EqFoutput['RTK.position'])[:,1]
RTK_pos = np.array(EqFoutput['RTK.position'])[:,0]
RTK_xyz = np.empty([3,len(RTKpostime)])
for i in range(len(RTK_pos)):
    lat = RTK_pos[i].x
    long = RTK_pos[i].y
    alt = RTK_pos[i].z
    RTK_xyz[:,i] = lla2xyz(lat,long,alt,earthrad) #Convert RKT lat lon alt to xyz coordinates
EqF_origin = lla2xyz(EqFoutput['EqF.origin'][0].x,EqFoutput['EqF.origin'][0].y,EqFoutput['EqF.origin'][0].z,earthrad)
RTK_rel = RTK_xyz.T - EqF_origin
R0_EqF = rotmat_xyz2NED(EqFoutput['EqF.origin'][0].x, EqFoutput['EqF.origin'][0].y)
RTK_NED_rel = R0_EqF @ RTK_rel.T

#Work out the M8 relative positions, and replacing the down component of position to the alt measurement from the baro
M8postime = np.array(EqFoutput['M8.position'])[:,1]
M8postime = np.array(list(M8postime), dtype=float)
M8_pos = np.array(EqFoutput['M8.position'])[:,0]

M8_relativepos = np.empty([3,len(M8postime)])
EqF_origin = EqFoutput['EqF.origin'][0]
for i in range(len(M8_pos)):
    x_pos = gps_diff(EqF_origin, M8_pos[i])
    M8_relativepos[:,i] = [x_pos.x, x_pos.y, x_pos.z]

baro = np.array(EqFoutput['plot.baro'])
baro = baro-baro[0] #correct to a relative baro measurement
barotime = np.array(EqFoutput['baro.time'])
#Interpolate baro measurements to match M8postime
c = np.searchsorted(M8postime, barotime[0])
d = np.searchsorted(M8postime,barotime[-1],side ='right')
def interp_func(x,y,newx):
    f = interp1d(x,y)
    newy = f(newx)
    return newy
baro_interp = interp_func(barotime,baro,M8postime[c:d])

M8xy_and_baro = np.empty([3,len(M8postime[c:d])])
M8xy_and_baro[0,:] = M8_relativepos[0,c:d]
M8xy_and_baro[1,:] = M8_relativepos[1,c:d]
M8xy_and_baro[2,:] = -baro_interp

#Plot EqF relative position vs the RTK relative position (used as ground truth) and also M8 GPS + baro position, origin is EqF GPS origin
RTKpostime = np.array(EqFoutput['RTK.position'])[:,1]
RTK_pos = np.array(EqFoutput['RTK.position'])[:,0]
RTK_relativepos = np.empty([3,len(RTKpostime)])
for i in range(len(RTK_pos)):
    x_p = gps_diff(EqF_origin, RTK_pos[i])
    RTK_relativepos[:,i] = [x_p.x, x_p.y, x_p.z]

fig1, ax = plt.subplots(3)
fig1.suptitle('Position plot using EqF')
ax[0].plot(EqFoutput['EqF.time'], EqFoutput['EqF.posx'], c = 'r')
ax[0].plot(RTKpostime, RTK_relativepos[0,:], c = 'b')
ax[0].plot(M8postime[c:d], M8xy_and_baro[0,:],'-.',c = 'darkgreen')
ax[0].set_ylabel('North (m)')
ax[1].plot(EqFoutput['EqF.time'], EqFoutput['EqF.posy'], c = 'r')
ax[1].plot(RTKpostime, RTK_relativepos[1,:], c = 'b')
ax[1].plot(M8postime[c:d], M8xy_and_baro[1,:],'-.',c = 'darkgreen')
ax[1].set_ylabel('East (m)')
ax[2].plot(EqFoutput['EqF.time'], EqFoutput['EqF.posz'], c = 'r',label='EqF Estimated Position')
ax[2].plot(RTKpostime, RTK_relativepos[2,:],  c = 'b',label = 'RTK GPS Position')
ax[2].plot(M8postime[c:d], M8xy_and_baro[2,:], '-.', c = 'darkgreen', label = 'M8 GPS and baro for alt')
ax[2].legend(loc='best',prop={'size': 8})
ax[2].set_xlabel('Time (s)')
ax[2].set_ylabel('Down (m)')
plt.setp(ax, xlim=[0,EqFoutput['EqF.time'][-1]])
figure = plt.gcf()  # get current figure
figure.set_size_inches(8, 5) # set figure's size manually to your full screen
plt.savefig('Estimated position ' + args.log + '.pdf', bbox_inches='tight') # bbox_inches removes extra white spaces

#Plot 3D position of EqF, RTK GPS position (referred to as ground truth), M8GPS and baro position (measurements used in EqF)
fig = plt.figure()
ax = plt.axes(projection='3d')
a = np.where((RTKpostime>0) & (RTKpostime<(EqFoutput['EqF.time'][-1])))
ax.plot3D(RTK_relativepos[0,a[0]], RTK_relativepos[1,a[0]], RTK_relativepos[2,a[0]], label = 'RTK GPS position', c = 'b')
b = np.where((np.array(EqFoutput['EqF.time'])>(RTKpostime[0])) & (np.array(EqFoutput['EqF.time'])<(EqFoutput['EqF.time'][-1])))
ax.plot3D(np.array(EqFoutput['EqF.posx'])[b[0]], np.array(EqFoutput['EqF.posy'])[b[0]], np.array((EqFoutput['EqF.posz']))[b[0]],'r',label='EqF Estimated Position')
f = np.where((M8postime[c:d]>0) & (M8postime[c:d]<(EqFoutput['EqF.time'][-1])))
ax.plot3D(M8xy_and_baro[0,f[0]],M8xy_and_baro[1,f[0]],M8xy_and_baro[2,f[0]], '-.', c = 'darkgreen', label = 'M8 GPS position and baro for alt')
ax.plot3D(np.array(EqFoutput['EqF.posx'])[b[0]][0], np.array(EqFoutput['EqF.posy'])[b[0]][0], np.array((EqFoutput['EqF.posz']))[b[0]][0], 'mo', label='Starting Position', linewidth = 7)
ax.plot3D(np.array(EqFoutput['EqF.posx'])[b[0]][-1], np.array(EqFoutput['EqF.posy'])[b[0]][-1], np.array((EqFoutput['EqF.posz']))[b[0]][-1], 'o',c = 'black', label='Finishing Position', linewidth = 7)
ax.set_xlabel('North (m)')
ax.set_ylabel('East (m)')
ax.set_zlabel('Down (m)')
ax.legend(loc='best',prop={'size': 8})
plt.title('3D position plot using EqF')


#Plot EqF velocity vs the RTK velocity (used as ground truth)
RTKtime = np.array(EqFoutput['RTK.velocity'])[:,3]
RTK_vel = np.array(EqFoutput['RTK.velocity'])[:,0:3]

fig2, ax2 = plt.subplots(3)
fig2.suptitle('Velocity plot using EqF')
ax2[0].plot(EqFoutput['EqF.time'], EqFoutput['EqF.velx'], c = 'r')
ax2[0].plot(RTKtime, RTK_vel[:,0],'b')
ax2[0].set_ylabel('North ($ms^{-1}$)')
ax2[1].plot(EqFoutput['EqF.time'], EqFoutput['EqF.vely'], c = 'r')
ax2[1].plot(RTKtime, RTK_vel[:,1],'b')
ax2[1].set_ylabel('East ($ms^{-1}$)')
ax2[2].plot(EqFoutput['EqF.time'], EqFoutput['EqF.velz'], c = 'r', label='EqF Estimated Velocity')
ax2[2].plot(RTKtime, RTK_vel[:,2],'b', label = 'RTK GPS Velocity')
ax2[2].legend(loc='best')
ax2[2].set_xlabel('Time (s)')
ax2[2].set_ylabel('Down ($ms^{-1}$)')
plt.setp(ax2, xlim=[0,EqFoutput['EqF.time'][-1]])
plt.setp(ax2, ylim=[-5,5])
figure = plt.gcf()  # get current figure
figure.set_size_inches(8, 5) # set figure's size manually to your full screen
plt.savefig('Estimated velocity ' + args.log + '.pdf', bbox_inches='tight') # bbox_inches removes extra white spaces