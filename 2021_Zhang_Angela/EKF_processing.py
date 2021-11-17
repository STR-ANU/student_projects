#!/usr/bin/env python

'''
Processes the EKF(3) output from SITL logs
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


    output = { 'EK3.Roll' : [],
               'EK3.Pitch' : [],
               'EK3.Yaw' : [],
               'EK3.north' :[],
               'EK3.east' :[],
               'EK3.down' :[],
               'EK3org.lat':[],
               'EK3org.long':[],
               'EK3org.alt':[],
               'EK3.time':[],
               'EKF.magtime':[],
               'EK3.magN':[],
               'EK3.magE':[],
               'EK3.magD':[],
               'EK3.biasgx':[],
               'EK3.biasgy':[],
               'EK3.biasgz':[],
               'XKF.time':[],
               'EK3.biasax':[],
               'EK3.biasay':[],
               'EK3.biasaz':[],
               'IVAC.time':[],
               'INAC.time':[],
               'Inno.PNPE':[],
               'InnoVar.PNPE':[],  
               'IVAD.time':[],
               'INAD.time':[],
               'Inno.PD':[],
               'InnoVar.PD':[],  
               'EKF.GPSnis':[],
               'EKF.altnis':[],
               'SVAA.time':[],
               'SVAA.state':[],
               'SVAB.state':[],
               'EKF.Covariances':[],
               'EK3.VN':[],
               'EK3.VE':[],
               'EK3.VD':[],
               'EK3.dtaverage':[],
               'EKF.GPSinnovsquared':[]
                }

    RGPI = None
    RFRH = None
    P = np.zeros((24,24))
    
    globaltime = None
    EKForigin_set = None
    timestep_min = None
    timestep_max = None
    altvariancetime = None

    while True:
        # we want replay sensor data, plus EKF3 result and SITL data
        m = mlog.recv_match(type=['XKF1','XKF2','XKF3','RFRH','RFRF','RISH','RISI','RGPH','RGPI','RGPJ','RFRH','RBRH','RBRI','RMGH','RMGI','ORGN',
                                  'SVAA','SVAB','XKV1','XKV2','CVAA','CVAB','CVAC','CVAD','CVAE','CVAF','CVAG','CVAH','CVAI','CVAJ', #24 raw states, state variance and covariances
                                  'CVAK','CVAL','CVAM','CVAN','CVAO','CVAP','CVAQ','CVAR','CVAS',
                                  'INAA','INAB','INAC','INAD','INAE','IVAA','IVAB','IVAC','IVAD','IVAE',
                                  'XKT'
                                  ])
        if m is None:
            break
        t = m.get_type()
        
        if t == 'XKT': #find timestep average, save min and max value
            timestep_min = m.EKFMin
            timestep_max = m.EKFMax
        
        if t == 'ORGN' and EKForigin_set is None: #EKF origin
            output['EK3org.lat'].append(m.Lat)
            output['EK3org.long'].append(m.Lng)
            output['EK3org.alt'].append(m.Alt)
            EKForigin_set = 1

        if t == 'XKF1' and m.C == 0:
            # output attitude, position, velocity and gyro bias of first EKF3 lane
            tsec = m.TimeUS*1.0e-6
            output['EK3.Roll'].append((tsec, m.Roll))
            output['EK3.Pitch'].append((tsec, m.Pitch))
            output['EK3.Yaw'].append((tsec, m.Yaw))
            output['EK3.time'].append(tsec)
            output['EK3.north'].append(m.PN)
            output['EK3.east'].append(m.PE)
            output['EK3.down'].append(m.PD)
            output['EK3.VN'].append(m.VN)
            output['EK3.VE'].append(m.VE)
            output['EK3.VD'].append(m.VD)
            output['EK3.biasgx'].append(m.GX)
            output['EK3.biasgy'].append(m.GY)
            output['EK3.biasgz'].append(m.GZ)
            
        if t == 'XKF2' and m.C == 0:
            # output accel bias of first EKF3 lane
            tsec = m.TimeUS*1.0e-6
            output['XKF.time'].append(tsec)
            output['EK3.biasax'].append(m.AX)
            output['EK3.biasay'].append(m.AY)
            output['EK3.biasaz'].append(m.AZ)
            if np.linalg.norm(np.array([m.MN,m.ME,m.MD]))!= 0: #used for checking mag readings, for debug purpose
                output['EKF.magtime'].append(tsec)
                N = m.MN/np.linalg.norm(np.array([m.MN,m.ME,m.MD]))
                E = m.ME/np.linalg.norm(np.array([m.MN,m.ME,m.MD]))
                D = m.MD/np.linalg.norm(np.array([m.MN,m.ME,m.MD]))
                output['EK3.magN'].append(N)
                output['EK3.magE'].append(E)
                output['EK3.magD'].append(D)          
            
        if t == 'RFRH':
            # replay frame header, used for timestamp of the frame
            RFRH = m

        if t == 'RGPI' and m.I == 0:
            # GPS0 status info, remember it so we know if we have a 3D fix
            RGPI = m
            
        if t == 'IVAC' and m.PN is not None and m.PE is not None:
            #extract variance of innovation vector from horizontal (NE) position
            tsec = m.TimeUS*1.0e-6
            output['IVAC.time'].append(tsec)
            innov = [m.PN, m.PE]
            output['InnoVar.PNPE'].append(innov)
            
        if t == 'INAC':
            #Extract horizontal position innovation
            tsec = m.TimeUS*1.0e-6
            output['INAC.time'].append(tsec)
            IPNPE = [m.PN, m.PE]
            output['Inno.PNPE'].append(IPNPE)
            
        if t == 'IVAD' and m.PD is not None and m.PD !=0:
            #Extract vertical position innovation
            tsec = m.TimeUS*1.0e-6
            altvariancetime = tsec
            output['IVAD.time'].append(tsec)
            output['InnoVar.PD'].append(m.PD)
            
        if t == 'INAD' and altvariancetime == m.TimeUS*1.0e-6:
            #Vertical position innovation (altitude)
            tsec = m.TimeUS*1.0e-6
            output['INAD.time'].append(tsec)
            output['Inno.PD'].append(m.PD)
            
        if t == 'SVAA' and EKForigin_set == 1:
            #Fetch the first 15 states
            raw_states = []
            for i in range(15):
                raw_states.append(getattr(m,"%02u"%(i)))
                
            output['SVAA.state'].append(raw_states)
            #delta angle bias in rad, delta velocity bias in m/s
            output['SVAA.time'].append(m.TimeUS*1.0e-6)
            globaltime = m.TimeUS*1.0e-6
            
        if t == 'SVAB' and EKForigin_set == 1:
            #Fetch the 16the state
            raw_state = getattr(m,"%02u"%(15))
            output['SVAB.state'].append(raw_state)
            
        if t == 'XKV1' and EKForigin_set == 1:
            #Fill in state variance (for first 12 states)
            assert m.TimeUS*1.0e-6 == globaltime, "XKV1 time does not match with the raw EKF state time"
            for i in range(12):
                P[i][i] = getattr(m,"V%02u"%(i)) 
            debug = 1
                
        if t == 'XKV2' and EKForigin_set == 1:
            #State variance for the last 12 states
            assert m.TimeUS*1.0e-6 == globaltime, "XKV2 time does not match with the raw EKF state time"
            for i in range(12,24):
                P[i][i] = getattr(m,"V%02u"%(i))
            debug = 1
            
        if t.startswith('CVA') and EKForigin_set == 1:
            #Fill in rest of the state covariance matrix
            assert m.TimeUS*1.0e-6 == globaltime, "Covariance time does not match with the raw EKF state time"
            # print(t)
            idx = ord(t[3]) - ord('A')
            vec = []
            vlen = 15
            if t == 'CVAS':
                vlen = 6
            for i in range(vlen):
                vec.append(getattr(m,"%02u"%(i+1)))

            elstart = idx * 15
            elnum = 0
            for x in range(24):
                for y in range(24):
                    if x >= y:
                        continue
                    if elnum >= elstart and elnum < elstart + 15:
                        P[x][y] = vec[elnum - elstart]
                    if (elnum-elstart) >= 15:
                        break
                    elnum += 1
                if (elnum-elstart) >= 15:
                    break
            if t == 'CVAS':
                P = P + P.T - np.diag(np.diag(P))
                output['EKF.Covariances'].append(P)
                P = np.zeros((24,24))

    
    assert (timestep_min + timestep_max)/2 != 0, "time step average is 0"
    dtaverage = (timestep_min + timestep_max)/2
    output['EK3.dtaverage'].append(dtaverage)    

    #Calculate horizontal (GPS) position NIS
    inno = np.array(output['Inno.PNPE'])
    innoVar = np.array(output['InnoVar.PNPE'])
    assert len(output['INAC.time'])==len(output['IVAC.time']), "time don't match for INAC and IVAC"
    inno_time = np.array(output['INAC.time'])
    
    for i in range(len(inno_time)):
        innovation_squared = (np.reshape(inno[i,:],(1,2)) @ inno[i,:]).item()
        output['EKF.GPSinnovsquared'].append(innovation_squared)
        
        GPSnis = (np.reshape(inno[i,:],(1,2))@ (np.linalg.inv(np.eye(2) * innoVar[i,:])) @ inno[i,:])[0]
        output['EKF.GPSnis'].append(GPSnis)
    
    #Calculate vertical (altitude) position NIS
    inno = np.array(output['Inno.PD'])
    innoVar = np.array(output['InnoVar.PD'])
    assert len(output['INAD.time'])==len(output['IVAD.time']), "time don't match for INAD and IVAD"
    inno_time = np.array(output['INAD.time'])
    
    for i in range(len(inno_time)):
        assert innoVar[i] != 0, "Alt variance is 0 it is %r" %innoVar[i]
        altnis = inno[i] * (1/ innoVar[i]) * inno[i]
        output['EKF.altnis'].append(altnis)
        
    import pickle
    #Output states are stored in pckl file
    outputfile = 'EKFoutput_' + filename + '.pckl'
    f = open(outputfile, 'wb')
    pickle.dump(output, f)
    f.close() 
    
estimate(args.log)


