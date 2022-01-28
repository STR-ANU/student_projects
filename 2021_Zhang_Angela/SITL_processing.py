#!/usr/bin/env python

'''
Processing SITL outputs from SITL logs
'''
from __future__ import print_function
from builtins import range

import os

from MAVProxy.modules.lib import mp_util
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("log", metavar="LOG")
parser.add_argument("--debug", action='store_true')

args = parser.parse_args()

from pymavlink import mavutil
from pymavlink.rotmat import Vector3, Matrix3
from pymavlink.mavextra import expected_earth_field_lat_lon
import math
from math import degrees

GRAVITY_MSS = 9.80665

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


def rot_to_euler(rot):
    '''find Euler angles (321 convention) for the matrix'''
    if rot[2][0] >= 1.0:
        pitch = pi
    elif rot[2][0] <= -1.0:
        pitch = -pi
    else:
        pitch = -math.asin(rot[2][0])
    roll = math.atan2(rot[2][1], rot[2][2])
    yaw  = math.atan2(rot[1][0], rot[0][0])
    return (roll, pitch, yaw)

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

def estimate(filename):
    '''run estimator over a replay log'''
    print("Processing log %s" % filename)

    mlog = mavutil.mavlink_connection(filename)

    output = { 'SIM.Roll' : [],
               'SIM.Pitch' : [],
               'SIM.Yaw' : [],
               'SIM.lat' :[],
               'SIM.long' :[],
               'SIM.alt' :[],
               'SIM.time':[],
               'SIM.Xi_R':[],
               'SIM.quaternion':[],
               'SIM2.time':[],
               'SIM2.Xi_p':[],
               'SIM2.Xi_v':[]
                }

    RGPI = None
    RFRH = None


    while True:
        # we want replay sensor data, plus EKF3 result and SITL data
        m = mlog.recv_match(type=['XKF1','XKF2','XKF3','SIM','SIM2','RFRH','RFRF','RISH','RISI','RGPH','RGPI','RGPJ','RFRH','RBRH','RBRI','RMGH','RMGI','ORGN',
                                  'IVAA','IVAB','IVAC','IVAD','CVAA','CVAB','CVAC','CVAD','CVAE','CVAF','CVAG','CVAH','CVAI','CVAJ',
                                  'CVAK','CVAL','CVAM','CVAN','CVAO','CVAP','CVAQ','CVAR','CVAS','CVAT'])
        if m is None:
            break
        t = m.get_type()
              

        if t == 'SIM':
            # output SITL attitude
            tsec = m.TimeUS*1.0e-6
            output['SIM.Roll'].append((tsec, m.Roll))
            output['SIM.Pitch'].append((tsec, m.Pitch))
            output['SIM.Yaw'].append((tsec, m.Yaw))
            
            #Convert roll, pitch, yaw to rotation matrices
            rot = euler_to_rot(m.Roll, m.Pitch, m.Yaw)
            output['SIM.Xi_R'].append(rot)
            
            #Output absolute position
            output['SIM.time'].append(tsec)
            output['SIM.lat'].append(m.Lat)
            output['SIM.long'].append(m.Lng)
            output['SIM.alt'].append(m.Alt)
            
            #output SITL quaternion
            quat = [m.Q1, m.Q2, m.Q3, m.Q4]
            output['SIM.quaternion'].append(quat)

        if t == 'SIM2':
            tsec = m.TimeUS*1.0e-6
            output['SIM2.time'].append(tsec)
            output['SIM2.Xi_p'].append(np.array([[m.PN],[m.PE],[m.PD]]))
            output['SIM2.Xi_v'].append(np.array([[m.VN],[m.VE],[m.VD]]))
            
        if t == 'RFRH':
            # replay frame header, used for timestamp of the frame
            RFRH = m

        if t == 'RGPI' and m.I == 0:
            # GPS0 status info, remember it so we know if we have a 3D fix
            RGPI = m

    import pickle

    outputfile = 'SITLoutput_' + filename + '.pckl'
    f = open(outputfile, 'wb')
    pickle.dump(output, f)
    f.close() 
    
estimate(args.log)


