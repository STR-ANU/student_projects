#!/usr/bin/env python

'''
Estimate attitude, position and velocity from an ArduPilot replay log using EqF
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

from pymavlink import mavutil
from pymavlink.rotmat import Vector3, Matrix3
from pymavlink.mavextra import expected_earth_field_lat_lon
import math
from math import degrees

import pickle

GRAVITY_MSS = 9.80665
param = args.param
#Fetch default noise parameters used in EKF3
if param == 'true':
    import subprocess
    command = 'python mav_param_test.py '+ args.log
    subprocess.call(command, stdout=open('EKFparams ' + args.log +'.txt','w'))
    print("using params in", args.log)
    import pandas as pd
    df = pd.read_csv('EKFparams ' + args.log + '.txt', sep='\s+', header= None)
    param_list = ['EK3_VELNE_M_NSE','EK3_VELD_M_NSE','EK3_POSNE_M_NSE',
                  'EK3_ALT_M_NSE','EK3_MAG_M_NSE','EK3_GYRO_P_NSE',
                  'EK3_ACC_P_NSE']
    par_dict = {}
    for parm in param_list:
        value = df.loc[df[0] == parm, 1].item()
        par_dict[parm] = value
        
    print(par_dict)

import numpy as np
from scipy.linalg import expm
from scipy.linalg import logm
from math import pi
import random
import utm

from scipy.interpolate import interp1d

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from statistics import mean

def SE23_inv(R,p,v):
    #inverse of SE23 element
    return R.T, -R.T @ p, -R.T @ v

def SE23_from_mat(mat):
    #Extract rotation, position and velocity from SE23 element
    R = mat[0:3,0:3]
    p = mat[0:3,3:4]
    v = mat[0:3,4:5]
    return R, p, v

def compute_skew_x (mat):
    #Compute skew matrix from vector
    skew_cross = np.array([(0, -mat[2], mat[1]),
                           (mat[2], 0, -mat[0]),
                           (-mat[1], mat[0], 0)])
    return skew_cross

def skew_to_vec (skew):
    return np.array([[skew[2][1]],[skew[0][2]],[skew[1][0]]])

def vect2se23 (vect):
    #Reconstruct se23 from vector containing rotation, position and velocity
    u_R = vect[0:3,0]
    u_p = vect[3:6,0:1]
    u_v = vect[6:9,0:1]
    u_Rx = compute_skew_x(u_R)
    
    mat = np.block([
        [u_Rx, u_p, u_v ],
        [np.zeros((2,5))]
        ])
    return mat

def construct_se23(u_Rx, u_p, u_v):
    #Reconstruct se23 from rotation matrix, position and velocity vector
    mat = np.zeros((5,5))
    mat[0:3,0:3] = u_Rx
    mat[0:3,3:4] = u_p
    mat[0:3,4:5] = u_v
    return mat

def construct_Ahat(R_Ahat, p_Ahat, v_Ahat):
    #Construcut SE23 from rotation matrix, position and velocity vector
    mat = np.zeros((5,5))
    mat[0:3,0:3] = R_Ahat
    mat[0:3,3:4] = p_Ahat
    mat[0:3,4:5] = v_Ahat
    mat[3,3] = 1
    mat[4,4] = 1
    return mat

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
    return Vector3(dx, dy, dz)

def rot_to_euler(rot):
    #see https://www.geometrictools.com/Documentation/EulerAngles.pdf for conversions and orders
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

class EqF:
    def __init__(self,gravity, param):
            self.imutime = 0
            self.gpstime = 0
            self.barotime = 0
            #Process noise should be tuned in later versions
            self.P = np.diag(np.concatenate((0.0001*np.ones(3), 0.0003*np.ones(3), 0.0001*np.ones(3), 0.00001*np.ones(6))))
            self.g = np.array([[0],[0],[gravity]])
            self.Ahat = np.eye(5)
            self.bhat = np.zeros((6,1))
            self.P_0 = np.eye(5)
            self.Riccati = np.diag(np.concatenate((0.01*np.ones(9), 0.01*np.ones(6))))
            self.identity = np.eye(9+6)
            self.NIS = 0
            if param == 'false':
                self.Q_posNED = 0.1 *np.eye(3)
                self.Q_posNE = 0.1 * np.eye(2)
                self.Q_gpsvel = 0.1 * np.eye(3)
                self.Q_posvel = 0.1 * np.eye(6)
                self.Q_baro = 0.1
                self.Q_mag = 0.1 * np.eye(3)
                
                self.R = 0.1*np.eye(6) #Noise introduced by input
                print("Default param set")
            else:
                self.Q_posNED = np.square(np.eye(3) * np.array([par_dict.get('EK3_POSNE_M_NSE'), par_dict.get('EK3_POSNE_M_NSE'), par_dict.get('EK3_ALT_M_NSE')]))
                self.Q_posNE = np.square(np.eye(2) * np.array([par_dict.get('EK3_POSNE_M_NSE'), par_dict.get('EK3_POSNE_M_NSE')]))
                self.Q_gpsvel = np.square(np.eye(3) * np.array([par_dict.get('EK3_VELNE_M_NSE'), par_dict.get('EK3_VELNE_M_NSE'), par_dict.get('EK3_VELD_M_NSE')]))
                self.Q_posvel =  np.square(np.eye(6) * np.array([par_dict.get('EK3_POSNE_M_NSE'), par_dict.get('EK3_POSNE_M_NSE'), par_dict.get('EK3_ALT_M_NSE'),par_dict.get('EK3_VELNE_M_NSE'), par_dict.get('EK3_VELNE_M_NSE'), par_dict.get('EK3_VELD_M_NSE')]))
                self.Q_baro = np.square(par_dict.get('EK3_ALT_M_NSE')) #separate output covariance to gps's self.Q
                self.Q_mag = np.square(np.eye(3) * 1000* np.array([par_dict.get('EK3_MAG_M_NSE'), par_dict.get('EK3_MAG_M_NSE'), par_dict.get('EK3_MAG_M_NSE')])) #times 1000 because the noise is specified in Gauss, measurement in milliGauss
                print("Mag noise is: ", 1000* par_dict.get('EK3_MAG_M_NSE'))
    
                self.R = np.square(np.eye(6) * np.array([par_dict.get('EK3_GYRO_P_NSE'), par_dict.get('EK3_GYRO_P_NSE'), par_dict.get('EK3_GYRO_P_NSE'), par_dict.get('EK3_ACC_P_NSE'), par_dict.get('EK3_ACC_P_NSE'), par_dict.get('EK3_ACC_P_NSE')])) #Noise introduced by input
                print("Using EKF parameter")
                
    @staticmethod
    def construct_A(g):
        #Construct input matrix
        A = np.zeros((9,9))
        A[6:9,0:3] = compute_skew_x(g.ravel())
        A[3:6,6:9] = np.eye(3)
        return A
    
    @staticmethod    
    def construct_C_GPS_posNE(p_Ahat):
        #constructs C matrix for GPS excluding the alt measurement
        C_gps = np.zeros((3,9))
        C_gps[:,0:3] = compute_skew_x(-p_Ahat.ravel())
        C_gps[:,3:6] = np.eye(3)
        mat = np.array([[1,0,0],[0,1,0]])
        C_gps_NE = mat @ C_gps
        return C_gps_NE
    
    @staticmethod    
    def construct_C_GPS_posvel(p_Ahat,v_Ahat):
        #constructs C matrix for GPS NED position and velocity
        C_gpsNED = np.zeros((6,9))
        C_gpsNED[0:3,0:3] = compute_skew_x(-p_Ahat.ravel())
        C_gpsNED[0:3,3:6] = np.eye(3)
        C_gpsNED[3:6,0:3] = compute_skew_x(-v_Ahat.ravel())
        C_gpsNED[3:6,6:9] = np.eye(3)
        return C_gpsNED
    
    @staticmethod    
    def construct_C_GPS_posNED(p_Ahat):
        #constructs C matrix for GPS NED position
        C_gps = np.zeros((3,9))
        C_gps[:,0:3] = compute_skew_x(-p_Ahat.ravel())
        C_gps[:,3:6] = np.eye(3)
        return C_gps
    
    def construct_C_GPS_vel(v_Ahat):
        #construct C matrix for GPS velocity
        C_gpsvel = np.zeros((3,9))
        C_gpsvel[:,0:3] = compute_skew_x(-v_Ahat.ravel())
        C_gpsvel[:,6:9] = np.eye(3)
        return C_gpsvel
    
    @staticmethod
    def construct_C_baro(p_Ahat):
        #construct C matrix for barometer (alt measurements)
        C_gps = np.zeros((3,9))
        C_gps[:,0:3] = compute_skew_x(-p_Ahat.ravel())
        C_gps[:,3:6] = np.eye(3)
        mat = np.array([0,0,-1])
        C_baro = mat @ C_gps
        return C_baro
    
    @staticmethod
    def construct_C_mag(R_Ahat, e1):
        #Construct C matrix for magnetometer
        C_mag = np.zeros((3,9))
        e1_skew = compute_skew_x(e1.ravel())
        C_mag[:,0:3] = R_Ahat.T @ e1_skew
        return C_mag   
    
    def construct_B(Ahat):
        #Construct input matrix
        B = np.zeros((9,6))
        R,p,v = SE23_from_mat(Ahat)
        R_T,p_T,v_T = SE23_from_mat(Ahat.T)
        p_cross = compute_skew_x((-R.T @ p).ravel())
        v_cross = compute_skew_x((-R.T @ v).ravel())
        B[0:3,0:3] = R
        B[3:6,0:3] = -R @ p_cross
        B[6:9,0:3] = -R @ v_cross
        B[6:9,3:6] = R
        return B
    
    def stateEstimate(self): #Project Ahat to estimated state
        return self.P_0 @ self.Ahat
    
    def rightaction(A,P): #General right action
        R_P, p_P, v_P = SE23_from_mat(P)
        R_A, p_A, v_A = SE23_from_mat(A)
        
        action_R = R_P @ R_A
        action_p = p_P + R_P @ p_A
        action_v = v_P + R_P @ v_A
        
        return construct_se23( action_R, action_p, action_v)
        
    def lift(self,gyro,accel):
        Phat = self.stateEstimate()
        R_Phat, p_Phat, v_Phat = SE23_from_mat(Phat)
        
        lift_R = compute_skew_x(gyro)
        lift_p = R_Phat.T @ v_Phat
        lift_v = accel + R_Phat.T @ self.g
        
        return construct_se23(lift_R, lift_p, lift_v)
    
    @staticmethod
    def construct_statemat(A,B): #state matrix after considering gyro & accel bias
        statemat = np.zeros((15,15))
        statemat[0:9,0:9] = A
        statemat[0:9,9:15] = -B
        return statemat
    
    @staticmethod
    def construct_inputmat(B): #input matrix after considering gyro & accel bias
        inputmat = np.zeros((15,6))
        inputmat[0:9,0:6] = B
        return inputmat
    
    @staticmethod
    def outputmat_gps_NE(C_gps): #Use when using gps horizontal (north & east) position
        outputmat = np.zeros((2,15))
        outputmat[0:2,0:9] = C_gps
        return outputmat
    
    @staticmethod
    def outputmat_gps_NED(C_gps): #use when using gps horizontal and vertical position
        outputmat = np.zeros((3,15))
        outputmat[0:3,0:9] = C_gps
        return outputmat
    
    @staticmethod
    def outputmat_gps_posvel(C_gps): #use when using gps position and velocity measurements concurrently
        outputmat = np.zeros((6,15))
        outputmat[0:6,0:9] = C_gps
        return outputmat
    
    @staticmethod
    def outputmat_baro(C_baro): #use when using alt measurements
        outputmat = np.zeros((1,15))
        outputmat[:,0:9] = C_baro
        return outputmat
    
    @staticmethod
    def outputmat_mag(C_mag): #use when using mag measurements
        outputmat = np.zeros((3,15)) 
        outputmat[0:3,0:9] = C_mag
        return outputmat
    
    def Ahat_wo_innov(self,gyro,accel,currentimutime): #Predict step
        timestep = currentimutime - self.imutime
        self.imutime = currentimutime
        R_Ahat, p_Ahat, v_Ahat = SE23_from_mat(self.Ahat)
        A = EqF.construct_A(self.g)
        B = EqF.construct_B(self.Ahat)
        
        gyr_bar = np.reshape([gyro.x,gyro.y,gyro.z], (3,1)) - self.bhat[0:3,:]
        acc_bar = np.array([[accel.x],[accel.y],[accel.z]]) - self.bhat[3:6,:]
        
        Lift = self.lift(gyr_bar.ravel(), acc_bar)
        state_mat = EqF.construct_statemat(A, B)
        input_mat = EqF.construct_inputmat(B)

        self.Riccati = (np.eye(15) + timestep*state_mat) @ self.Riccati @ (np.eye(15) + timestep*state_mat).T + timestep*self.P + timestep*(input_mat @ self.R @ input_mat.T)
        
        # self.Ahat = self.Ahat @ expm(timestep*Lift)
        # Better discretization of the system dynamics updated Oct 11
        gyro_cross = compute_skew_x(gyr_bar.ravel())
        Ahatdot = self.Ahat @ Lift
        R_Ahatdot, p_Ahatdot, v_Ahatdot = SE23_from_mat(Ahatdot)
        
        R_Ahat_new = R_Ahat @ expm(timestep*gyro_cross)
        v_Ahat_new = v_Ahat + timestep * v_Ahatdot
        p_Ahat_new = p_Ahat + timestep * v_Ahat + (timestep**2/2) * v_Ahatdot
        self.Ahat = construct_Ahat(R_Ahat_new, p_Ahat_new, v_Ahat_new)
        
    def gpscorrect(self,x_p, currentgpstime):
        timestep = currentgpstime - self.gpstime
        self.gpstime = currentgpstime
        R_Ahat, p_Ahat, v_Ahat = SE23_from_mat(self.Ahat)
        
        #Uncomment this section if using GPS NE pos
        y = np.array([[x_p.x],[x_p.y]])
        C = EqF.construct_C_GPS_posNE(p_Ahat)
        output_mat = EqF.outputmat_gps_NE(C)
        S = output_mat @ self.Riccati @ output_mat.T + self.Q_posNE #Q changed to eye(2) because only NE measurements
        K = self.Riccati @ output_mat.T @ np.linalg.inv(S)
        Delta = self.Riccati @ output_mat.T @ np.linalg.inv(S) @(y-p_Ahat[0:2])
        
        self.Riccati = (self.identity - K @ output_mat) @ self.Riccati
        self.Ahat = expm(vect2se23(Delta[0:9,:])) @ self.Ahat
        self.bhat = self.bhat + Delta[9:15,:]
        self.NIS = ((y-p_Ahat[0:2]).T @ np.linalg.inv(S) @ (y-p_Ahat[0:2])).item()
        
        # #Uncomment this section if using GPS NED position measurement for correction
        # y = np.array([[x_p.x],[x_p.y],[x_p.z]])
        # C = EqF.construct_C_GPS_posNED(p_Ahat)
        # output_mat = EqF.outputmat_gps_NED(C)
        # S = output_mat @ self.Riccati @ output_mat.T + self.Q_posNED
        # K = self.Riccati @ output_mat.T @ np.linalg.inv(S)
        # Delta = self.Riccati @ output_mat.T @ np.linalg.inv(S) @(y-p_Ahat)
              
        # self.Riccati = (self.identity - K @ output_mat) @ self.Riccati
        # self.Ahat = expm(vect2se23(Delta[0:9,:])) @ self.Ahat
        # self.bhat = self.bhat + Delta[9:15,:]
        # self.NIS = ((y-p_Ahat).T @ np.linalg.inv(S) @ (y-p_Ahat)).item()
        
    def gpsvelcorrect(self,vel,currentgpstime): #update state using GPS velocity measurement
        timestep = currentgpstime - self.gpstime
        self.gpstime = currentgpstime
        R_Ahat, p_Ahat, v_Ahat = SE23_from_mat(self.Ahat)
        
        v = np.array([[vel.x],[vel.y],[vel.z]])
        C = EqF.construct_C_GPS_vel(v_Ahat)
        output_mat = EqF.outputmat_gps_NED(C)
        S = output_mat @ self.Riccati @ output_mat.T + self.Q_gpsvel
        K = self.Riccati @ output_mat.T @ np.linalg.inv(S)
        Delta = self.Riccati @ output_mat.T @ np.linalg.inv(S) @(v-v_Ahat)
              
        self.Riccati = (self.identity - K @ output_mat) @ self.Riccati
        self.Ahat = expm(vect2se23(Delta[0:9,:])) @ self.Ahat
        self.bhat = self.bhat + Delta[9:15,:]
        
        
    def barocorrect(self, alt): #update with alt measurement
        R_Ahat, p_Ahat, v_Ahat = SE23_from_mat(self.Ahat)
        
        C = EqF.construct_C_baro(p_Ahat)
        output_mat = EqF.outputmat_baro(C)
        S = output_mat @ self.Riccati @ output_mat.T + self.Q_baro
        K = self.Riccati @ output_mat.T @ np.linalg.inv(S)
        Delta = self.Riccati @ output_mat.T @ np.linalg.inv(S) @(alt-(-p_Ahat[2]))
        Delta = np.array([Delta]).T
              
        self.Riccati = (self.identity - K @ output_mat) @ self.Riccati
        self.Ahat = expm(vect2se23(Delta[0:9,:])) @ self.Ahat
        self.bhat = self.bhat + Delta[9:15,:]    
        
    def magcorrect(self, latlon, field): #update with mag measurement
        R_Ahat, p_Ahat, v_Ahat = SE23_from_mat(self.Ahat)
        mag = np.array([[field.x],[field.y],[field.z]])
        norm_mag = mag
        
        #Calculate expected mag field for initial location (NED)
        if latlon is None:
            return None, None
        earth_field = expected_earth_field_lat_lon(latlon.Lat*1.0e-7, latlon.Lon*1.0e-7)
        e1_earth = np.array([[earth_field.x],[earth_field.y],[earth_field.z]])
        
        C = EqF.construct_C_mag(R_Ahat, e1_earth)
        output_mat = EqF.outputmat_mag(C)
        S = output_mat @ self.Riccati @ output_mat.T + self.Q_mag
        K = self.Riccati @ output_mat.T @ np.linalg.inv(S)
        Delta = self.Riccati @ output_mat.T @ np.linalg.inv(S) @(norm_mag - R_Ahat.T @ e1_earth)
        
        self.Riccati = (self.identity - K @ output_mat) @ self.Riccati
        self.Ahat = expm(vect2se23(Delta[0:9,:])) @ self.Ahat
        self.bhat = self.bhat + Delta[9:15,:]
        
        R_Ahat, p_Ahat, v_Ahat = SE23_from_mat(self.Ahat)
        return R_Ahat @ norm_mag, e1_earth

class Estimator(object):
    '''state estimator'''
    def __init__(self):
        self.eqf = EqF(GRAVITY_MSS, param)
        self.origin = None
        self.baro_origin = None

    def update_GPS(self, position, velocity,currenttime):
        '''handle new GPS sample'''
        if self.origin is None:
            self.origin = position
        x_p = gps_diff(self.origin, position)
        self.eqf.gpscorrect(x_p, currenttime)
        self.eqf.gpsvelcorrect(velocity, currenttime)

    def update_MAG(self, latlon, field):
        '''handle new magnetometer sample'''
        norm_mag_inertial, e1 = self.eqf.magcorrect(latlon, field)
        if args.debug:
            print('MAG: ', field)
        return norm_mag_inertial, e1

    def update_BARO(self, altitude):
        '''handle new barometer sample'''
        if args.debug:
            print('BARO: ', altitude)
        if self.baro_origin is None:
            self.baro_origin = altitude #set baro first altitude measurement as origin
            if abs(self.baro_origin) < 1e-3:
                print('baro_origin is zero')
            else:
                print('Baro origin is ',self.baro_origin)
        alt = altitude - self.baro_origin
        self.eqf.barocorrect(alt) #here the altitude is the barometric alt above GPS origin in meters

    def update_IMU(self, delta_velocity, dv_dt, delta_angle, da_dt, timenow):
        '''handle new IMU sample'''

        accel = delta_velocity / dv_dt
        gyro = delta_angle / da_dt
        self.eqf.Ahat_wo_innov(gyro, accel, timenow)
    

def estimate(filename, param):
    '''run estimator over a replay log'''
    print("Processing log %s" % filename)

    mlog = mavutil.mavlink_connection(filename)

    est = Estimator()

    output = { 'EqF.Roll' : [],
               'EqF.Pitch' : [],
               'EqF.Yaw' : [],
               'EqF.posx':[],
               'EqF.posy':[],
               'EqF.posz':[],
               'EqF.velx':[],
               'EqF.vely':[],
               'EqF.velz':[],
               'EqF.time':[],
               'plot.baro':[],
               'baro.time':[],
               'plot.magtime':[],
               'plot.normmagN':[],
               'plot.normmagE':[],
               'plot.normmagD':[],
               'mag.earthN':[],
               'mag.earthE':[],
               'mag.earthD':[],
               'EqF.biasgx':[],
               'EqF.biasgy':[],
               'EqF.biasgz':[],
               'EqF.biasax':[],
               'EqF.biasay':[],
               'EqF.biasaz':[],
               'EqF.NIS':[],
               'EqF.GPStime':[],
               'EqF.R_inv':[],
               'EqF.p_inv':[],
               'EqF.v_inv':[],
               'EqF.Riccati':[],
               'EqF.bhat':[],
               'EqF.GPSpos':[],
               'EqF.GPSrotT':[],
               'Riccati.eig':[],
               'eigen.time':[],
               'eigenvect':[],
               'EqF.origin':[],
               'RTKorigin':[],
               'RTK.position':[],
               'RTK.velocity':[],
               'RTK.time':[],
               'M8.position':[]
                }

    RGPI = None
    RFRH = None
    latlon = None
    GPS_originset = None

    
    ##Uncomment if want to add additional bias to delta angle and delta velocity manually
    # bias_dang = np.array([[0.01],[0.02],[0.03]])
    # bias_dvel = np.array([[0.1],[0.2],[0.2]])
    # print('bias dang is',bias_dang)
    # print('bias dvel is',bias_dvel)
    
    # first gather GPS data
    GPS_Data = [[],[]]
    GPS_Dict = [{},{}]
    GPS_Status = [0,0]

    while True:
        # we want replay sensor data, plus EKF3 result and SITL data
        m = mlog.recv_match(type=['RFRH','RGPI', 'RGPJ'])
        if m is None:
            break
        t = m.get_type()

        if t == 'RFRH':
            # replay frame header, used for timestamp of the frame
            RFRH = m

        if t == 'RGPI' and m.I == args.gps_num:
            # GPS0 status info, remember it so we know if we have a 3D fix
            RGPI = m
        
        if t == 'RGPI' and m.I <=1:
            # remember status so we can tell if RTK fix
            GPS_Status[m.I] = m.Stat

        if t == 'RGPJ' and m.I <=1 and GPS_Status[m.I] >= 3:
            # update on GPS0, with 3D fix
            pos = Vector3(m.Lat*1.0e-7, m.Lon*1.0e-7, m.Alt*1.0e-2)
            if latlon is None:
                latlon = m
            vel = Vector3(m.VX, m.VY, m.VZ)
            tsec = RFRH.TimeUS*1.0e-6
            GPS_Data[m.I].append((pos, vel, tsec, GPS_Status[m.I]))

    # adjust GPS data for lag of half the sample rate
    for j in range(2):
        if len(GPS_Data[j]) == 0:
            continue
        for i in range(len(GPS_Data[j])-1):
            (pos1,vel1,tsec1,status1) = GPS_Data[j][i]
            (pos2,vel2,tsec2,status2) = GPS_Data[j][i+1]
            GPS_Dict[j][tsec1] = ((pos1+pos2)*0.5, (vel1+vel2)*0.5, min(status1,status2))

        (pos1,vel1,tsec1,status) = GPS_Data[j][-1]
        GPS_Dict[j][tsec1] = (pos1, vel1, status)

    # form RTK.position output only from RTK fixed data
    if len(GPS_Data[1]) == 0:
        print("There is only 1 GPS source, no RTK GPS data")
    for i in range(len(GPS_Data[1])-1):
        (pos,vel,tsec,status) = GPS_Data[1][i]
        if status != 6:
            continue
        if args.correct_gps:
            (pos,vel,status) = GPS_Dict[1][tsec]
        else:
            print("correct_gps value is false, not correcting RTK by half sample")
        output['RTK.velocity'].append((vel.x, vel.y, vel.z, tsec)) #Extracted actual velocity values from Vector3 for ease of plotting
        output['RTK.position'].append((pos, tsec))


    print("Processed %u/%u GPS samples" % (len(GPS_Dict[0].keys()), len(GPS_Dict[1].keys())))
    mlog.rewind()

    while True:
        # we want replay sensor data, plus EKF3 result and SITL data
        m = mlog.recv_match(type=['XKF1','XKF2','XKF3','SIM','SIM2','RFRH','RFRF','RISH','RISI',
                                  'RGPH','RGPI','RGPJ','RFRH','RBRH','RBRI','RMGH','RMGI','ORGN', 'ARM'])
        if m is None:
            break
        t = m.get_type()
               
        if t == 'ARM':
            #stop processing on disarm
            if m.ArmState == 0:
                print("Copter disarmed, stopped processing")
                break
            
        if t == 'RFRH':
            # replay frame header, used for timestamp of the frame
            RFRH = m

        if t == 'RGPI' and m.I == args.gps_num:
            # GPS0 status info, remember it so we know if we have a 3D fix
            RGPI = m

        if t == 'RGPJ' and m.I == args.gps_num and RGPI.Stat >= 3:
            # update on GPS0, with 3D fix
            pos = Vector3(m.Lat*1.0e-7, m.Lon*1.0e-7, m.Alt*1.0e-2)
            if latlon is None:
                latlon = m
            if GPS_originset is None:
                GPS_originset = 1
            vel = Vector3(m.VX, m.VY, m.VZ)
            tsec = RFRH.TimeUS*1.0e-6
            # get data from time corrected dictionay if available
            if args.correct_gps:
                (pos, vel, status) = GPS_Dict[m.I][tsec]
            est.update_GPS(pos, vel, tsec)
            output['M8.position'].append((pos, tsec))
            R_Ahat, p_Ahat, v_Ahat = SE23_from_mat(est.eqf.Ahat)
            R_inv, p_inv, v_inv = SE23_inv(R_Ahat,p_Ahat,v_Ahat)
            
            #NIS computation
            output['EqF.GPStime'].append(tsec)
            output['EqF.NIS'].append(est.eqf.NIS)
            
            #Output EqF Ahat_inverse
            output['EqF.GPSpos'].append(p_Ahat)
            output['EqF.GPSrotT'].append(R_Ahat.T)
            output['EqF.R_inv'].append(R_inv)
            output['EqF.p_inv'].append(p_inv)
            output['EqF.v_inv'].append(v_inv)
            
            #Required for NEES computation
            output['EqF.Riccati'].append(est.eqf.Riccati)
            output['EqF.bhat'].append(est.eqf.bhat)
            
            #Output eigen value of the riccati (for debugging, can ignore)
            val, vect = np.linalg.eig(est.eqf.Riccati)
            output['Riccati.eig'].append(val.real)
            if max(val) > 10:
                output['eigen.time'].append(tsec)
                output['eigenvect'].append(vect[:,val.argmax()])

        if t == 'RMGI' and m.I == 0 and m.H == 1:
            # first compass sample, healthy compass
            field = Vector3(m.FX, m.FY, m.FZ)
           
            #save earth magnetic field vs predicted earth magnetic field, used for debugging and can ignore
            norm_mag_inertial, mag = est.update_MAG(latlon, field)
            if (norm_mag_inertial is not None) and (mag is not None):
                N = norm_mag_inertial[0][0]
                output['plot.normmagN'].append(N)
                E = norm_mag_inertial[1][0]
                output['plot.normmagE'].append(E)
                D = norm_mag_inertial[2][0]
                output['plot.normmagD'].append(D)
                N = mag[0][0]
                output['mag.earthN'].append(N)
                E = mag[1][0]
                output['mag.earthE'].append(E)
                D = mag[2][0]
                output['mag.earthD'].append(D)
                tsec = RFRH.TimeUS*1.0e-6
                output['plot.magtime'].append(tsec)
            

        if t == 'RBRI' and m.I == 0:
            # first baro sample after GPS value is set
            if GPS_originset == 1:
                est.update_BARO(m.Alt)
                output['plot.baro'].append(m.Alt)
                tsec = RFRH.TimeUS*1.0e-6
                output['baro.time'].append(tsec)
            
        if t == 'RISI' and m.I == 0:
            # update on IMU0
            tsec = RFRH.TimeUS*1.0e-6
            dvel = Vector3(m.DVX, m.DVY, m.DVZ)
            ##Uncomment following if want to add additional bias to delta angle and delta velocity manually
            # dvel.x += bias_dvel[0,0] * m.DVDT
            # dvel.y += bias_dvel[1,0] * m.DVDT
            # dvel.z += bias_dvel[2,0] * m.DVDT
            dang = Vector3(m.DAX, m.DAY, m.DAZ)
            # dang.x += bias_dang[0,0] * m.DADT
            # dang.y += bias_dang[1,0] * m.DADT
            # dang.z += bias_dang[2,0] * m.DADT
            
            est.update_IMU(dvel, m.DVDT, dang, m.DADT, tsec)

            # output euler roll/pitch
            R_Ahat, p_Ahat, v_Ahat = SE23_from_mat(est.eqf.Ahat)
            r,p,y = rot_to_euler(R_Ahat)
            output['EqF.Roll'].append((tsec, degrees(r)))
            output['EqF.Pitch'].append((tsec, degrees(p)))
            output['EqF.Yaw'].append((tsec, wrap_360(degrees(y))))
            
            #Output eqf position
            output['EqF.time'].append(tsec)
            output['EqF.posx'].append(p_Ahat[0][0])
            output['EqF.posy'].append(p_Ahat[1][0])
            output['EqF.posz'].append(p_Ahat[2][0])
            
            #Output eqf velocituy
            output['EqF.velx'].append(v_Ahat[0][0])
            output['EqF.vely'].append(v_Ahat[1][0])
            output['EqF.velz'].append(v_Ahat[2][0])
            
            #Output eqf gyro bias
            output['EqF.biasgx'].append(est.eqf.bhat[0][0])
            output['EqF.biasgy'].append(est.eqf.bhat[1][0])
            output['EqF.biasgz'].append(est.eqf.bhat[2][0])
            output['EqF.biasax'].append(est.eqf.bhat[3][0])
            output['EqF.biasay'].append(est.eqf.bhat[4][0])
            output['EqF.biasaz'].append(est.eqf.bhat[5][0])
    
    output['EqF.origin'].append(est.origin)
    
    #Save file in pickle format
    outputfile = 'EqFoutput_' + filename + ' EKparam ' + param + '.pckl'
    f = open(outputfile, 'wb')
    pickle.dump(output, f)
    f.close()    
    
estimate(args.log, args.param)


