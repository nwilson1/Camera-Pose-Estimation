## By Noah Wilson
## MMAE 539
## 5/6/21

import numpy as np
from aux_funcs import *
import matplotlib.pyplot as plt
import os
import time

## This code implements a perspective based camera pose sensor in three contexts:
## finding the camera pose in 154 still images, finding pose of the camera
## as it rolls down an incline, and a particle filter implementation.

## The "Pictures" directory, "calibrate" directory, and the aux_funcs.py file must be
## present in the same directory as this file in order to run.

def main_pics():

    K,RealR = calibrate_camera() # Get calibration matrix and real point distances

    files = sorted(os.listdir('Pictures'))
    dpips = 0.8 # Distance between Lego pips on the board in cm
    Pboard = [1.9,36] # X and Z position of the left-front corner of the board relative to the top-right corner of the image in cm
    RealZs = dpips*np.arange(6,28,2) + Pboard[1] # Finds real Z values of the camera in each image
    RealXs = dpips*np.arange(0,28,2) + Pboard[0] # Finds real X values of the camera in each image
    RealXYZ = np.array([[(x,-8.1,z) for x in RealXs] for z in RealZs]).reshape(154,3)
    CalcXYZ = []

    start = time.time()
    for file in files: # Loops over files in the Pictures directory
        img = cv2.imread('Pictures/'+file)
        CalcXYZ.append(FindPose(img,K,RealR).T[0])
    CalcXYZ = np.array(CalcXYZ)

    fig1,axs = plt.subplots()
    axs.plot(RealXYZ[:,0],RealXYZ[:,2],'o')
    axs.plot(CalcXYZ[:,0],CalcXYZ[:,2],'C3x')
    axs.set_xlabel('X (cm)')
    axs.set_ylabel('Z (cm)')
    axs.set_xlim([0,25])
    axs.set_ylim([37,62])
    axs.invert_yaxis()
    axs.legend(['Real Poses','Measured Poses'],loc='upper center', bbox_to_anchor=(0.5, 1.1),
          fancybox=True, shadow=True, ncol=5)
    axs.set_aspect('equal')
    fig2,axs = plt.subplots()
    axs.plot(RealXYZ[:,1])
    axs.plot(CalcXYZ[:,1],'x')
    print(time.time()-start)
    plt.show()

def main_vid():
    ## There are three availalbe videos to analize, "Ramp1.mp4", "Ramp2.mp4", and "Ramp3.mp4",
    K,RealR = calibrate_camera() # Get calibration matrix and real point distances
    vobj = cv2.VideoCapture('Ramp1.mp4') # Change the file name here to analize a different video
    CalcXYZ = []
    start = time.time()
    cont,img = vobj.read()

    while cont: # Loops over all frames in the video
        CalcXYZ.append(FindPose(img,K,RealR).T[0])
        cont,img = vobj.read()
    t = np.linspace(0,2.2,len(CalcXYZ))
    print(time.time()-start)
    h = 7.0 # Elevation of the ramp
    L = 110.5 # Length of the ramp
    a_g = 980*h/L # Acceleration due to gravity on the ramp in cm/s^2
    a_f = 37 # Acceleration due to friction in cm/s^2. Determined emperically.
    a = a_g - a_f # Total acceleration down the ramp

    plt.plot(t,0.987*np.array(CalcXYZ)[0,2]+.5*a*t**2)
    plt.plot(t,0.987*np.array(CalcXYZ)[:,2],'.')
    plt.xlabel('time (s)')
    plt.ylabel(r'$Z$ (cm)')
    plt.title('Distance vs. Time on Ramp')
    plt.legend(['Real Z','Measured Z'])
    plt.show()

def main_PF():

    K,RealR = calibrate_camera() # Get calibration matrix and real point distances
    vobj = cv2.VideoCapture('Ramp3.mp4') # Change the file name here to analize a different video
    cont,img = vobj.read()
    mu = 0.987*FindPose(img,K,RealR) #Initial pose estimate [x y z]
    mu = np.append(mu,[0]).reshape(4,1)

    Sig = np.array([[100,   0,   0, 0],   #                         [xx xy xz]
                    [  0, 100,   0, 0],   # Initial pose covariance [yx yy yz]
                    [  0,   0, 100, 0],
                    [  0,   0,   0, 0]])  #                         [zx zy zz]

    Q = np.array([[100,   0,  0],
                  [  0, 100, 0],
                  [  0,   0, 9]]) # Measurement covariance

    Qinv = np.linalg.pinv(Q)
    Qdet = np.linalg.det(Q)
    h = 7.0 # Ramp elevation
    L = 110.5 # Length of the ramp
    a_g = 980*h/L # Acceleration due to gravity on the ramp in cm/s^2
    a_f = 37 # Acceleration due to friction in cm/s^2. Determined emperically.
    a = a_g - a_f # Total acceleration down the ramp
    dt = 2.2/60 # Timestep between frames

    t = 0

    def g(MU): # Define state transformation function using a discrete kinematic model
        x,y,z,vz = MU
        xp = x
        yp = y
        vzp = vz + a*dt
        zp = z + vzp*dt + 0.5*a*dt**2
        return np.array([xp,yp,zp,vzp])
    M = 10000 # Initial umber of particles
    P =  np.random.multivariate_normal(mu.T[0],Sig,size=M) # Create particles
    w = np.zeros((M,3)) # Initialize list of M weights

    plt.ion()
    fig,axs = plt.subplots(1,2,figsize=(11,5))
    particles1 = axs[0].scatter(P[:,0],P[:,2],s=10,color='k') # Plot prior distribution of particles
    realXZ, = axs[0].plot(14.9,mu[2],'rx')
    particles2 = axs[1].scatter(P[:,0],P[:,1],s=10,color='k') # Plot prior distribution of particles
    realXY, = axs[1].plot(14.9,-10,'rx')
    axs[0].set_ylabel(r'$Z$ (cm)')
    axs[0].set_xlabel(r'$X$ (cm)')
    axs[1].set_ylabel(r'$Y$ (cm)')
    axs[1].set_xlabel(r'$X$ (cm)')
    axs[0].set_xlim([-30,60])
    axs[0].set_ylim([0,100])
    axs[1].set_xlim([-30,60])
    axs[1].set_ylim([-50,40])
    axs[0].set_aspect('equal')
    axs[1].set_aspect('equal')
    axs[0].legend(['Real Pose','Particles'])
    axs[1].legend(['Real Pose','Particles'])
    axs[0].set_title(r'$X$ vs. $Z$')
    axs[1].set_title(r'$X$ vs. $Y$')
    axs[0].invert_yaxis()

    plt.draw()

    while cont: # Loops over all frames in the video
        Z = 0.987*FindPose(img,K,RealR).T[0]
        cont,img = vobj.read()

        for m in range(len(P)):
            P[m] = g(P[m]) # Predict new positions of particles
            w[m] = np.exp(-0.5*(Z-P[m,:3]).T@Qinv@(Z-P[m,:3]))/np.sqrt(Qdet*(2*np.pi)**3) # Compute weights for the particles

        w_norm = w/np.sum(w,axis=0) # Normalized weights

        P_bins = np.cumsum(w_norm,axis=0) # Create bins with indices corresponding to each particle
        idx = [] # Initialize list of sampled bin indexes

        for m in range(len(P)):
            n = np.random.random() # Uniform random number [0,1)
            binx = np.digitize(n,P_bins[:,0]) # Sample the x value
            biny = np.digitize(n,P_bins[:,1]) # Sample the y value
            binz = np.digitize(n,P_bins[:,2]) # Sample the z value
            idx.extend([binx,biny,binz]) # Append index list
        idx.extend(np.random.randint(0,len(P),10))
        idx = np.array(list(set(idx))) # Remove duplicate indices from resampling process
        P = P[idx] # Index the set of particles with the sampled indices
        w = w[idx]

        t += dt
        realXZ.set_data(14.9,mu[2]+.5*a*t**2)
        realXY.set_data(14.9,-10)
        particles1.set_offsets(P[:,np.array([0,2])])
        particles2.set_offsets(P[:,np.array([0,1])])
        fig.canvas.draw_idle()
        plt.pause(0.01)

if __name__ == '__main__':

    ## Uncomment the function you want to run.
    ## main_pics outputs plots of the real and measured poses for all 154 test pictures
    ## main_vid outputs plots of the real and measured poses for the camera motion on an incline
    ## main_PF applies the incline video capture as measurements in a particle filter

    # main_pics()
    # main_vid()
    main_PF()
