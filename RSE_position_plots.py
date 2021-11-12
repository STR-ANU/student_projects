#!/usr/bin/env python
##This is to determine how the relative positions will work, especially when 
#SIM gives lat/lon/alt and origin is different to others and origin set at different time

'''
estimate attitude from an ArduPilot replay log using a python state estimator
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
from scipy.linalg import norm
from math import pi
import random
import utm

from scipy.interpolate import interp1d

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
#plt.close("all")

from statistics import mean

if "nobias" in args.log:
    EqFbias = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
    EKFinternal_bias = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
    print("Using 0 for true bias")
else:
    EqFbias = np.array([[-0.01],[-0.02],[-0.03],[0.1],[0.2],[0.2]])
    dtaverage = 0.0025
    EKFinternal_bias = dtaverage * np.array([[-0.01],[-0.02],[-0.03],[0.1],[0.2],[0.2]])
    print("Setting true bias value")

def skew_to_vec (skew):
    return np.array([[skew[2][1]],[skew[0][2]],[skew[1][0]]])

def wrap_180(angle):
    if angle > 180:
        angle -= 360.0
    if angle < -180:
        angle += 360.0
    return angle

def wrap_360(angle):
    if angle > 360:
        angle -= 360.0
    if angle < 0:
        angle += 360.0
    return angle

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
    # if (dx > 0.001 or dy > 0.001 or dz > 0.001):
    #     a = 1
    return Vector3(dx, dy, dz)

def interp_func(x,y,newx):
    f = interp1d(x,y)
    newy = f(newx)
    return newy

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


# graph all the fields we've output
import matplotlib.pyplot as plt
datatype = ["Roll", "Pitch", "Yaw"]
fig, axs = plt.subplots(len(datatype),1)

#EK3 Yaw is -180 to 180, wrap around so between 0 and 360 degrees
output['EK3.Yaw']

for i in range(len(datatype)):
    for k in output.keys():        
        if datatype[i] in k:
            t = [ v[0] for v in output[k] ]
            if datatype[i] == "Yaw":
                y = [wrap_360(v[1]) for v in output[k]]
            else:
                y = [ v[1] for v in output[k] ]
            if 'SIM' in k:
                axs[i].plot(t, y, '-.g', label= 'Simulator Attitude')
            elif 'EqF' in k:
                axs[i].plot(t, y,'--', c = 'orangered', label= 'EqF Estimated Attitude')
            elif 'EK3' in k:
                axs[i].plot(t, y, c = 'mediumblue', label= 'EKF3 Estimated Attitude')
            maxt = t[-1]
plt.legend(loc='best')
fig.suptitle('Estimated Attitude Using EKF3 and EqF Against Simulator Attitude')
plt.setp(axs[-1], xlabel='Time (s)')
plt.setp(axs[0], ylabel='Roll (Degree)')
plt.setp(axs[1], ylabel='Pitch (Degree)')
plt.setp(axs[2], ylabel='Yaw (Degree)')
# axs[0].set_ylim([-180, 180])
# axs[1].set_ylim([-90, 90])
axs[2].set_ylim([0, 360])
# plt.setp(axs, xlim=[0,300])
plt.show()
from pathlib import Path
Path("Graphs for simulation set 2/"+args.log).mkdir(parents=True, exist_ok=True)
figure = plt.gcf()  # get current figure
figure.set_size_inches(8, 5) # set figure's size manually to your full screen (32x18)
plt.savefig("Graphs for simulation set 2/"+ args.log+"/Attitude estimate " + args.log + ".pdf", bbox_inches='tight')

##Compute the 'Rotation RSE'
def euler_to_rot(roll, pitch, yaw):
    '''fill the matrix from Euler angles in **DEGREES** '''
    roll = math.radians(roll)
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    sr = math.sin(roll)
    cr = math.cos(roll)
    sy = math.sin(yaw)
    cy = math.cos(yaw)
    
    rot = np.zeros((3,3))
    rot[0][0] = cp * cy
    rot[0][1] = (sr * sp * cy) - (cr * sy)
    rot[0][2] = (cr * sp * cy) + (sr * sy)
    rot[1][0] = cp * sy
    rot[1][1] = (sr * sp * sy) + (cr * cy)
    rot[1][2] = (cr * sp * sy) - (sr * cy)
    rot[2][0] = -sp
    rot[2][1] = sr * cp
    rot[2][2] = cr * cp
    return rot

sim_rpy = []
EKF_rpy = []
EqF_rpy = []
    
for i in range(len(output['SIM.Roll'])):
    rpy = [output['SIM.Roll'][i][1],output['SIM.Pitch'][i][1], wrap_360(output['SIM.Yaw'][i][1])]
    sim_rpy.append(rpy)
for i in range(len(output['EK3.Roll'])):
    rpy = [output['EK3.Roll'][i][1],output['EK3.Pitch'][i][1], wrap_360(output['EK3.Yaw'][i][1])]
    EKF_rpy.append(rpy)
for i in range(len(output['EqF.Roll'])):
    rpy = [output['EqF.Roll'][i][1],output['EqF.Pitch'][i][1], wrap_360(output['EqF.Yaw'][i][1])]
    EqF_rpy.append(rpy)
sim_rpy = np.array(sim_rpy)
EKF_rpy = np.array(EKF_rpy)
EqF_rpy = np.array(EqF_rpy)
#Get EKF and EqF to match time with SIM
w = np.searchsorted(output['SIM.time'], output['EK3.time'][0])
x = np.searchsorted(output['SIM.time'],output['EK3.time'][-1],side ='right')
EKF_rpy_interp = np.empty([3, len(output['SIM.time'][w:x])])
for index in range(3):
    EKF_rpy_interp[index,:] = interp_func(output['EK3.time'],EKF_rpy[:,index],output['SIM.time'][w:x])
EKF_rot_rse = []
sim_rpy_4EKF = sim_rpy[w:x]
for index in range(len(sim_rpy_4EKF)):
    rot_true = euler_to_rot(sim_rpy_4EKF[index][0], sim_rpy_4EKF[index][1], sim_rpy_4EKF[index][2])
    rot_est = euler_to_rot(EKF_rpy_interp[0][index], EKF_rpy_interp[1][index], EKF_rpy_interp[2][index])
    rot_tilde = rot_true.T @ rot_est
    log_rot_tilde_vector = skew_to_vec(logm(rot_tilde))
    mag_angle_rse = norm(log_rot_tilde_vector)
    EKF_rot_rse.append(mag_angle_rse)
    
y = np.searchsorted(output['SIM.time'], output['EqF.time'][0])
z = np.searchsorted(output['SIM.time'],output['EqF.time'][-1],side ='right')
EqF_rpy_interp = np.empty([3, len(output['SIM.time'][y:z])])
for index in range(3):
    EqF_rpy_interp[index,:] = interp_func(output['EqF.time'],EqF_rpy[:,index],output['SIM.time'][y:z])
EqF_rot_rse = []
sim_rpy_4EqF = sim_rpy[y:z]
for index in range(len(sim_rpy_4EqF)):
    rot_true = euler_to_rot(sim_rpy_4EqF[index][0], sim_rpy_4EqF[index][1], sim_rpy_4EqF[index][2])
    rot_est = euler_to_rot(EqF_rpy_interp[0][index], EqF_rpy_interp[1][index], EqF_rpy_interp[2][index])
    rot_tilde = rot_true.T @ rot_est
    log_rot_tilde_vector = skew_to_vec(logm(rot_tilde))
    mag_angle_rse = norm(log_rot_tilde_vector)
    EqF_rot_rse.append(mag_angle_rse)

fig18 = plt.figure()
#convert the rotation rse to degrees before plotting
EqF_rot_rse = np.array(EqF_rot_rse)*180/pi
EKF_rot_rse = np.array(EKF_rot_rse)*180/pi
plt.plot(output['SIM.time'][w:x], EqF_rot_rse,'--', c="orangered", label = 'EqF')
plt.plot(output['SIM.time'][y:z], EKF_rot_rse, c="mediumblue", label = 'EKF3')
plt.yscale("log")
plt.legend(loc='best')
plt.title('Rotation angle error')
plt.ylim(bottom=0)
# plt.xlim(left=0, right=300)
plt.xlabel('Time (s)')
plt.ylabel('Angle error (Degrees)')
plt.show()
figure = plt.gcf()  # get current figure
figure.set_size_inches(8, 5) # set figure's size manually to your full screen (32x18)
fig18.savefig('Graphs for simulation set 2/'+ args.log+'/Angle error' + args.log + '.pdf', bbox_inches='tight')

#print the average EqF and EKF rotation angle error
start_t = 75
end_t = 175
eqf_flying_time = np.logical_and(np.array(output['SIM.time'][w:x])>=start_t, np.array(output['SIM.time'][w:x])<=end_t)
avg_eqf_rot_err = mean(EqF_rot_rse[eqf_flying_time])
ekf_flying_time = np.logical_and(np.array(output['SIM.time'][y:z])>=start_t, np.array(output['SIM.time'][y:z])<=end_t)
avg_ekf_rot_err = mean(EKF_rot_rse[ekf_flying_time])
print("Averages on %s log" %filename)
print("EqF average angle error is:" ,avg_eqf_rot_err)
print("EKF average angle error is:", avg_ekf_rot_err, "for t=[", start_t,end_t,"]")


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

# fig9, ax9 = plt.subplots(3)
# fig9.suptitle('Estimated position with baro+mag on nolag2 log')
# # ax9[0].plot(output['EK3.time'], EK3_N)
# ax9[0].plot(output['SIM.time'], pos_rel[0,:],'-.')
# ax9[0].plot(output['SIM2.time'], sim2_N)
# # ax9[0].plot(output['EqF.time'], EqF_N, ':')
# ax9[0].set_ylabel('North (m)')
# # ax9[1].plot(output['EK3.time'], EK3_E)
# ax9[1].plot(output['SIM.time'], pos_rel[1,:],'-.')
# ax9[1].plot(output['SIM2.time'], sim2_E)
# # ax9[1].plot(output['EqF.time'], EqF_E, ':')
# ax9[1].set_ylabel('East (m)')
# # ax9[2].plot(output['EK3.time'],EK3_D,label='EK3')
# ax9[2].plot(output['SIM.time'], pos_rel[-1,:],'-.',label='SITL using lla to ned')
# ax9[2].plot(output['SIM2.time'], sim2_D, label='SITL from SIM2')
# # ax9[2].plot(output['EqF.time'], EqF_D, ':', label='EqF')
# ax9[2].legend(loc='best')
# ax9[2].set_xlabel('Time(s)')
# ax9[2].set_ylabel('Down (m)')
    
#Calculate origin of EKF3 in x,y,z coordinate
EK3_org = lla2xyz(output['EK3org.lat'][0], output['EK3org.long'][0], output['EK3org.alt'][0], earthrad)
R0_EK3 = rotmat_xyz2NED(output['EK3org.lat'][0], output['EK3org.long'][0])

diff = R0_EK3 @ EK3_org #- R0 @ p[:,0]
EK3_N = output['EK3.north'] + diff[0]
EK3_E = output['EK3.east'] + diff[1]
EK3_D = output['EK3.down'] + diff[2]

EqF_org = lla2xyz(output['EqF.origin'][0].x,output['EqF.origin'][0].y,output['EqF.origin'][0].z,earthrad)
R0_EqF = rotmat_xyz2NED(output['EqF.origin'][0].x, output['EqF.origin'][0].y)

diff2 = R0_EqF @ EqF_org #- R0 @ p[:,0]
EqF_N = output['EqF.posx'] + diff2[0]
EqF_E = output['EqF.posy'] + diff2[1]
EqF_D = output['EqF.posz'] + diff2[2]

####
# diff3 = R0_EqF @ EqF_org - R0 @ p[:,0]
# pos_rel = (pos_rel.T + diff3).T
# print('size of diff3 is ', diff3.shape)

#RMSE of position, using interpolation to match size


i = np.searchsorted(output['SIM.time'], output['EK3.time'][0])
l = np.searchsorted(output['SIM.time'],output['EK3.time'][-1],side ='right')
EK3_N_new =interp_func(output['EK3.time'],EK3_N,output['SIM.time'][i:l])
EK3_E_new =interp_func(output['EK3.time'],EK3_E,output['SIM.time'][i:l])
EK3_D_new =interp_func(output['EK3.time'],EK3_D,output['SIM.time'][i:l])
rse = np.sqrt(np.square(p_rel[0,i:l] - EK3_N_new) + np.square(p_rel[1,i:l] - EK3_E_new) + 
              np.square(p_rel[2,i:l] - EK3_D_new))

j = np.searchsorted(output['EqF.time'],output['SIM.time'][0])
k = np.searchsorted(output['EqF.time'],output['SIM.time'][-1],side ='right')
sitl_N =interp_func(output['SIM.time'],p_rel[0,:],output['EqF.time'][j:k])
sitl_E =interp_func(output['SIM.time'],p_rel[1,:],output['EqF.time'][j:k])
sitl_D =interp_func(output['SIM.time'],p_rel[2,:],output['EqF.time'][j:k])
eqf_rse = np.sqrt(np.square(sitl_N - EqF_N[j:k]) + np.square(sitl_E - EqF_E[j:k]) + 
              np.square(sitl_D - EqF_D[j:k]))

# plt.figure() # Here's the part I need
# plt.plot(output['EqF.time'][j:k],(eqf_rse), c="orangered", label = 'EqF')
# plt.plot(output['SIM.time'][i:l], (rse),c="mediumblue", label = 'EK3')
# plt.yscale("log")
# plt.legend(loc='best')
# plt.title('NED estimated positions RSE for %s using WMM, using lla converted position as ground truth' %filename)
# plt.xlabel('Time(s)')
# plt.ylabel('m')
# plt.show()

###Relative position plots
SIM_xyz = lla2xyz(output['SIM.lat'],output['SIM.long'],np.array(output['SIM.alt']),earthrad) #Convert SIM to xyz coordinates
#Interpolate to match EqF time
n = np.searchsorted(output['EqF.time'],output['SIM.time'][0])
o = np.searchsorted(output['EqF.time'],output['SIM.time'][-1],side ='right')
SIM_xyz_interp = np.empty([3, len(output['EqF.time'][n:o])])
for index in range(3):
    SIM_xyz_interp[index,:] = interp_func(output['SIM.time'],SIM_xyz[index,:],output['EqF.time'][n:o])
EqF_org_xyz = lla2xyz(output['EqF.origin'][0].x,output['EqF.origin'][0].y,output['EqF.origin'][0].z,earthrad)
SIM_rel = SIM_xyz_interp.T - EqF_org_xyz
R0_EqF = rotmat_xyz2NED(output['EqF.origin'][0].x, output['EqF.origin'][0].y)
SIM_NED_rel = R0_EqF @ SIM_rel.T

#Interpolate to match EKF time
a = np.searchsorted(output['EK3.time'],output['SIM.time'][0])
b = np.searchsorted(output['EK3.time'],output['SIM.time'][-1],side ='right')
SIM_xyz_interp2 = np.empty([3, len(output['EK3.time'][a:b])])
for index in range(3):
    SIM_xyz_interp2[index,:] = interp_func(output['SIM.time'],SIM_xyz[index,:],output['EK3.time'][a:b])
EKF_org_xyz = lla2xyz(output['EK3org.lat'][0],output['EK3org.long'][0],output['EK3org.alt'][0],earthrad)
SIM_rel_2 = SIM_xyz_interp2.T - EKF_org_xyz
R0_EKF = rotmat_xyz2NED(output['EK3org.lat'][0], output['EK3org.long'][0])
SIM_NED_relative = R0_EKF @ SIM_rel_2.T
EK3_PN = output['EK3.north']
EK3_PE = output['EK3.east']
EK3_PD = output['EK3.down']
EKF_rse_1 = np.sqrt(np.square(SIM_NED_relative[0,:] - EK3_PN[a:b]) + np.square(SIM_NED_relative[1,:] - EK3_PE[a:b]) + np.square(SIM_NED_relative[2,:] - EK3_PD[a:b]))

#Extract EqF relative PN PE PD
EqF_PN = output['EqF.posx']
EqF_PE = output['EqF.posy']
EqF_PD = output['EqF.posz']

eqf_rse_1 = np.sqrt(np.square(SIM_NED_rel[0,:] - EqF_PN[n:o]) + np.square(SIM_NED_rel[1,:] - EqF_PE[n:o]) + np.square(SIM_NED_rel[2,:] - EqF_PD[n:o]))

# plt.figure() ##!! Here's the RSE using the relative positions
# plt.plot(output['EqF.time'][n:o],(eqf_rse_1), c="orangered", label = 'EqF')
# plt.plot(output['EK3.time'][a:b],(EKF_rse_1), c="mediumblue", label = 'EK3')
# plt.yscale("log")
# plt.legend(loc='best')
# plt.title('NED estimated positions RSE for %s using relative NED positions' %filename)
# plt.xlabel('Time(s)')
# plt.ylabel('m')
# plt.show()

###Position RSE using the GPS diff function################
SITL_time = np.array(output['SIM.time'])
SIM_lat = np.array(output['SIM.lat'])
SIM_long = np.array(output['SIM.long'])
SIM_alt = np.array(output['SIM.alt'])
SIM_relativepos_EqF = np.empty([3,len(SITL_time)])
#Find SITL relative position with respect to EqF origin
for i in range(len(SITL_time)):
    SIM_position = Vector3(SIM_lat[i], SIM_long[i], SIM_alt[i])
    x_p = gps_diff(output['EqF.origin'][0], SIM_position)
    SIM_relativepos_EqF[:,i] = [x_p.x, x_p.y, x_p.z]

#Calculate eqf rse
# #Method 1: use EqF time    
# u = np.searchsorted(output['EqF.time'],output['SIM.time'][0])
# v = np.searchsorted(output['EqF.time'],output['SIM.time'][-1],side ='right')
# SIM_relativepos_interp = np.empty([3, len(output['EqF.time'][u:v])])
# #Interpolate SITL relative positions to match up with EqF time
# for i in range(3):
#     SIM_relativepos_interp[i,:] = interp_func(output['SIM.time'],SIM_relativepos_EqF[i,:],output['EqF.time'][u:v])
# #Extract EqF relative PN PE PD
# EqF_PN = output['EqF.posx']
# EqF_PE = output['EqF.posy']
# EqF_PD = output['EqF.posz']
# eqf_rse_2 = np.sqrt(np.square(SIM_relativepos_interp[0,:] - EqF_PN[u:v]) + np.square(SIM_relativepos_interp[1,:] - EqF_PE[u:v]) 
#                     + np.square(SIM_relativepos_interp[2,:] - EqF_PD[u:v]))

#Method 2: use EKF time
u = np.searchsorted(output['EK3.time'],output['SIM.time'][0])
v = np.searchsorted(output['EK3.time'],output['SIM.time'][-1],side ='right')
SIM_relativepos_interp = np.empty([3, len(output['EK3.time'][u:v])])
for i in range(3):
    SIM_relativepos_interp[i,:] = interp_func(output['SIM.time'],SIM_relativepos_EqF[i,:],output['EK3.time'][u:v])
EqF_PN_interp = interp_func(output['EqF.time'],output['EqF.posx'],output['EK3.time'][u:v])
EqF_PE_interp = interp_func(output['EqF.time'],output['EqF.posy'],output['EK3.time'][u:v])
EqF_PD_interp = interp_func(output['EqF.time'],output['EqF.posz'],output['EK3.time'][u:v])
eqf_rse_2 = np.sqrt(np.square(SIM_relativepos_interp[0,:] - EqF_PN_interp) + np.square(SIM_relativepos_interp[1,:] - EqF_PE_interp) + np.square(SIM_relativepos_interp[2,:] - EqF_PD_interp))
eqf_PNerror = abs(SIM_relativepos_interp[0,:] - EqF_PN_interp)
eqf_PEerror = abs(SIM_relativepos_interp[1,:] - EqF_PE_interp)
eqf_PDerror = abs(SIM_relativepos_interp[2,:] - EqF_PD_interp)


# #Plot the SITL interpolated position with EQF interpolated position
# fig15, ax15 = plt.subplots(3)
# ax15[0].plot(output['EK3.time'][u:v], SIM_relativepos_interp[0,:])
# ax15[0].plot(output['EK3.time'][u:v], EqF_PN_interp)
# fig15.suptitle('EqF position vs True position for %s' %filename)
# ax15[0].set_title('Position North')
# ax15[1].plot(output['EK3.time'][u:v], SIM_relativepos_interp[1,:])
# ax15[1].plot(output['EK3.time'][u:v], EqF_PE_interp)
# ax15[1].set_title('Position East')
# ax15[2].plot(output['EK3.time'][u:v], SIM_relativepos_interp[2,:],label='EqF estimated positions')
# ax15[2].plot(output['EK3.time'][u:v], EqF_PD_interp,label='SITL position')
# ax15[2].set_title('Position Down')
# ax15[2].legend(loc='best')
# ax15[2].set_xlabel('Time(s)')
# ax15[2].set_ylabel('m')
# figure = plt.gcf()  # get current figure
# figure.set_size_inches(8, 5) # set figure's size manually to your full screen (32x18)
# fig15.savefig('Graphs for simulation set 2/'+ args.log+'/True position ' + args.log + '.pdf', bbox_inches='tight')

#Calculate EKF rse
SIM_relativepos_EKF = np.empty([3,len(SITL_time)])
for i in range(len(SITL_time)):
    EKF_origin = Vector3(output['EK3org.lat'][0],output['EK3org.long'][0],output['EK3org.alt'][0])
    SIM_position = Vector3(SIM_lat[i], SIM_long[i], SIM_alt[i])
    x_p = gps_diff(EKF_origin, SIM_position)
    SIM_relativepos_EKF[:,i] = [x_p.x, x_p.y, x_p.z]
    
x = np.searchsorted(output['EK3.time'],output['SIM.time'][0])
y = np.searchsorted(output['EK3.time'],output['SIM.time'][-1],side ='right')
SIM_relpos_interp = np.empty([3, len(output['EK3.time'][x:y])])
for i in range(3):
    SIM_relpos_interp[i,:] = interp_func(output['SIM.time'],SIM_relativepos_EKF[i,:],output['EK3.time'][x:y])
EK3_PN = output['EK3.north']
EK3_PE = output['EK3.east']
EK3_PD = output['EK3.down']
EKF_rse_2 = np.sqrt(np.square(SIM_relpos_interp[0,:] - EK3_PN[x:y]) + np.square(SIM_relpos_interp[1,:] - EK3_PE[x:y]) + np.square(SIM_relpos_interp[2,:] - EK3_PD[x:y]))

EK3_PNerror = abs(SIM_relpos_interp[0,:] - EK3_PN[x:y])
EK3_PEerror = abs(SIM_relpos_interp[1,:] - EK3_PE[x:y])
EK3_PDerror = abs(SIM_relpos_interp[2,:] - EK3_PD[x:y])


plt.figure() ##!! Here's the RSE using GPS diff function
plt.plot(output['EK3.time'][u:v],(eqf_rse_2),'--', c="orangered", label = 'EqF')
plt.plot(output['EK3.time'][x:y], EKF_rse_2, c="mediumblue", label = 'EKF3')
plt.yscale("log")
plt.legend(loc='best')
plt.title('Estimated Position Root-square-error (RSE)')
plt.ylim(bottom = 0.001, top = 1)
plt.xlim(left=70, right=190)
plt.xlabel('Time (s)')
plt.ylabel('Root-square-error (RSE) (m)')
plt.show()
figure = plt.gcf()  # get current figure
figure.set_size_inches(8, 5) # set figure's size manually to your full screen (32x18)
plt.savefig('Graphs for simulation set 2/'+ args.log+'/Position RSE ' + args.log + '.pdf', bbox_inches='tight')

#print the average EqF and EKF position RSE
start_t = 75
end_t = 175
eqf_flying_time = np.logical_and(np.array(output['EK3.time'][u:v])>=start_t, np.array(output['EK3.time'][u:v])<=end_t)
avg_eqf_pos_rse = mean(eqf_rse_2[eqf_flying_time])
ekf_flying_time = np.logical_and(np.array(output['EK3.time'][x:y])>=start_t, np.array(output['EK3.time'][x:y])<=end_t)
avg_ekf_pos_rse = mean(EKF_rse_2[ekf_flying_time])
print("EqF average position RSE is:" ,avg_eqf_pos_rse)
print("EKF average position RSE is:", avg_ekf_pos_rse, "for t=[", start_t,end_t,"]")

#Split the position error to different components
fig16, ax16 = plt.subplots(3)
ax16[0].plot(output['EK3.time'][u:v], eqf_PNerror,'--', c="orangered")
ax16[0].plot(output['EK3.time'][x:y], EK3_PNerror,c="mediumblue")
fig16.suptitle('Estimated Position Absolute Error')
ax16[0].set_ylabel('North (m)')
ax16[0].set_ylim(bottom=0, top = 0.3)
ax16[0].set_xlim([70,190])
ax16[1].plot(output['EK3.time'][u:v], eqf_PEerror,'--', c="orangered")
ax16[1].plot(output['EK3.time'][x:y], EK3_PEerror, c="mediumblue")
ax16[1].set_ylabel('East (m)')
ax16[1].set_ylim(bottom=0, top = 0.3)
ax16[1].set_xlim([70,190])
ax16[2].plot(output['EK3.time'][u:v], eqf_PDerror,'--', c="orangered",label='EqF Position Error')
ax16[2].plot(output['EK3.time'][u:v], EK3_PDerror, c="mediumblue",label='EKF Position Error')
ax16[2].set_ylabel('Down (m)')
ax16[2].set_ylim(bottom=0, top = 0.15)
ax16[2].set_xlim([70,190])
ax16[2].legend(loc='best')
ax16[2].set_xlabel('Time(s)')
figure = plt.gcf()  # get current figure
figure.set_size_inches(8, 5) # set figure's size manually to your full screen (32x18)
fig16.savefig('Graphs for simulation set 2/'+ args.log+'/Position split ' + args.log + '.pdf', bbox_inches='tight')

############################################################

# #Calculate origin of EKF3 in x,y,z coordinate
# EK3_org = lla2xyz(output['EK3org.lat'][0], output['EK3org.long'][0], output['EK3org.alt'][0], earthrad)
# R0_EK3 = rotmat_xyz2NED(output['EK3org.lat'][0], output['EK3org.long'][0])

# org_diff = R0_EK3 @ EK3_org - R0 @ p[:,0]
# EK3_N_diff = output['EK3.north'] + org_diff[0]
# EK3_E_diff = output['EK3.east'] + org_diff[1]
# EK3_D_diff = output['EK3.down'] + org_diff[2]

##Plot RSE for velocity error
# #Method 1: use EqF time 
# q = np.searchsorted(output['EqF.time'],output['SIM2.time'][0])
# r = np.searchsorted(output['EqF.time'],output['SIM2.time'][-1],side ='right')
# SITL_v = np.empty([3, len(output['EqF.time'][q:r])])
# for i in range(3):
#     SITL_v[i,:] = interp_func(np.array(output['SIM2.time']),np.array(output['SIM2.Xi_v'])[:,i,0], np.array(output['EqF.time'][q:r]))
# EqF_velN = output['EqF.velx'][q:r]
# EqF_velE = output['EqF.vely'][q:r]
# EqF_velD = output['EqF.velz'][q:r]
# eqf_vel_rse = np.sqrt(np.square(SITL_v[0,:] - EqF_velN) + np.square(SITL_v[1,:] - EqF_velE) + np.square(SITL_v[2,:] - EqF_velD))

#Method 2: use EKF time
q = np.searchsorted(output['EK3.time'],output['SIM2.time'][0])
r = np.searchsorted(output['EK3.time'],output['SIM2.time'][-1],side ='right')
SITL_v = np.empty([3, len(output['EqF.time'][q:r])])
for i in range(3):
    SITL_v[i,:] = interp_func(output['SIM2.time'],np.array(output['SIM2.Xi_v'])[:,i,0],output['EK3.time'][q:r])
EqF_velN = interp_func(output['EqF.time'],output['EqF.velx'],output['EK3.time'][q:r])
EqF_velE = interp_func(output['EqF.time'],output['EqF.vely'],output['EK3.time'][q:r])
EqF_velD = interp_func(output['EqF.time'],output['EqF.velz'],output['EK3.time'][q:r])
eqf_vel_rse = np.sqrt(np.square(SITL_v[0,:] - EqF_velN) + np.square(SITL_v[1,:] - EqF_velE) + np.square(SITL_v[2,:] - EqF_velD))
eqf_VNerror = abs(SITL_v[0,:] - EqF_velN)
eqf_VEerror = abs(SITL_v[1,:] - EqF_velE)
eqf_VDerror = abs(SITL_v[2,:] - EqF_velD)


s = np.searchsorted(output['EK3.time'],output['SIM2.time'][0])
t = np.searchsorted(output['EK3.time'],output['SIM2.time'][-1],side ='right')
SITL_v = np.empty([3, len(output['EK3.time'][s:t])])
for i in range(3):
    SITL_v[i,:] = interp_func(np.array(output['SIM2.time']),np.array(output['SIM2.Xi_v'])[:,i,0], np.array(output['EK3.time'][s:t]))
ekf_velN = output['EK3.VN'][s:t]
ekf_velE = output['EK3.VE'][s:t]
ekf_velD = output['EK3.VD'][s:t]
ekf_vel_rse = np.sqrt(np.square(SITL_v[0,:] - ekf_velN) + np.square(SITL_v[1,:] - ekf_velE) + np.square(SITL_v[2,:] - ekf_velD))
EK3_VNerror = abs(SITL_v[0,:] - ekf_velN)
EK3_VEerror = abs(SITL_v[1,:] - ekf_velE)
EK3_VDerror = abs(SITL_v[2,:] - ekf_velD)

#RSE for velocity
plt.figure()
plt.plot(output['EK3.time'][q:r],eqf_vel_rse, '--', c="orangered", label = 'EqF')
plt.plot(output['EK3.time'][s:t],ekf_vel_rse, c="mediumblue", label = 'EKF3')     
plt.yscale("log")
plt.legend(loc='best')
plt.title('Estimated Velocity Root-square-error (RSE)' )
plt.ylim(bottom=0.001, top = 1)
plt.xlim(left=70, right=190)
plt.xlabel('Time (s)')
plt.ylabel('Root-square-error (RSE) ($ms^{-1}$)')
plt.show()
figure = plt.gcf()  # get current figure
figure.set_size_inches(8, 5) # set figure's size manually to your full screen (32x18)
plt.savefig('Graphs for simulation set 2/'+ args.log+'/Velocity RSE ' + args.log + '.pdf', bbox_inches='tight')

#print the average EqF and EKF velocity RSE
start_t = 75
end_t = 175
eqf_flying_time = np.logical_and(np.array(output['EK3.time'][q:r])>=start_t, np.array(output['EK3.time'][q:r])<=end_t)
avg_eqf_vel_rse = mean(eqf_vel_rse[eqf_flying_time])
ekf_flying_time = np.logical_and(np.array(output['EK3.time'][s:t])>=start_t, np.array(output['EK3.time'][s:t])<=end_t)
avg_ekf_vel_rse = mean(ekf_vel_rse[ekf_flying_time])
print("EqF average velocity RSE is:" ,avg_eqf_vel_rse)
print("EKF average velocity RSE is:", avg_ekf_vel_rse, "for t=[", start_t,end_t,"]")


#Split velocity error to different components
fig17, ax17 = plt.subplots(3)
ax17[0].plot(output['EK3.time'][u:v], eqf_VNerror,'--', c="orangered",label='EqF Velocity Error')
ax17[0].plot(output['EK3.time'][x:y], EK3_VNerror, c="mediumblue",label='EKF Velocity Error')
ax17[0].set_xlabel('Time(s)')
ax17[0].legend(loc='best')
fig17.suptitle('Estimated Velocity Absolute Error')
ax17[0].set_ylabel('North ($ms^{-1}$)')
ax17[0].set_ylim(bottom=0, top = 0.15)
ax17[0].set_xlim([70,190])
ax17[1].plot(output['EK3.time'][u:v], eqf_VEerror,'--', c="orangered")
ax17[1].plot(output['EK3.time'][x:y], EK3_VEerror, c="mediumblue")
ax17[1].set_ylabel('East ($ms^{-1}$)')
ax17[1].set_ylim(bottom=0, top = 0.2)
ax17[1].set_xlim([70,190])
ax17[2].plot(output['EK3.time'][u:v], eqf_VDerror,'--', c="orangered",label='EqF Velocity Error')
ax17[2].plot(output['EK3.time'][u:v], EK3_VDerror, c="mediumblue",label='EKF Velocity Error')
ax17[2].set_ylabel('Down ($ms^{-1}$)')
ax17[2].legend(loc='best')
ax17[2].set_xlabel('Time(s)')
ax17[2].set_ylim(bottom=0, top =0.1)
ax17[2].set_xlim([70,190])
figure = plt.gcf()  # get current figure
figure.set_size_inches(8, 5) # set figure's size manually to your full screen (32x18)
fig17.savefig('Graphs for simulation set 2/'+ args.log+'/Velocity split ' + args.log + '.pdf', bbox_inches='tight')