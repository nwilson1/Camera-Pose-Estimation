import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import os

def findRs(Ps):
    ## Finds the distance beteween each point of an array and every other point
    Rs = []
    idxs = np.arange(len(Ps)) # List of indices for the input array
    for i in range(len(Ps)):
        Rs.append(np.sqrt(np.sum((Ps[i] - Ps[idxs!=i])**2,axis=1)))
    return np.array(Rs)

def sortpts(Pts):
    ## Sorts the pixel position of points in the image by both X and Y (U and V)
    UV = Pts[Pts[:,0].argsort()] # Sort by U value
    for i in np.arange(8,88,8):
        UV[i-8:i] = UV[UV[i-8:i,1].argsort()+i-8] # Sort by V value
    UVW = np.concatenate((UV,np.ones((len(UV),1))),1)#.T # Add w value to the array
    return UVW

def getpts(Img):
    ## Finds the pixel position of all 80 corners in the checkerboard pattern in an image
    grayscale = cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
    Pts = np.array(cv2.goodFeaturesToTrack(grayscale,80,0.02,10))[:,0]
    return Pts

def Calc_XY(Z,K,uv):
    ## Calculate the x and y position os the points given a guessed value
    ## of Z and a camera calibration matrix.
    xy = (np.linalg.inv(K) @ (Z*uv))
    return xy

def FindPose(Img,K,RealR):
    ## Determines the pose of the camera relative to the top-left forner
    ## of the image.  Uses the camera matrix and real point distances
    ## from calibrate_camera.

    C = np.array([[640,360,1]]).T # Pixel center of the image
    pts = getpts(Img)
    uvw = sortpts(pts).T # Pixel positons of the corners in the image

    def Err(Zguess):
        ## Finds the error between the real distances between points and the
        ## distances between the calculated points.
        XY = Calc_XY(Zguess,K,uvw)
        CalcR = findRs(XY.T)

        return np.sqrt(np.sum((RealR - CalcR)**2)) # Error computed as a Frobenius norm

    Z = minimize_scalar(Err).x # Find distance to the checkerboard by minimizing the error

    Origin = Calc_XY(Z,K,uvw[:,0]).T
    Image_Center = Calc_XY(Z,K,C).T

    Cam_Pose = np.round((np.array([1,-1,1])*(Image_Center - Origin + np.array([0,0,Z]))).T,1)
    return Cam_Pose

def calibrate_camera():

    ## Finds camera calibration matrix and the real distances between corners of the checkerboard
    z = 0.8*np.arange(6,28,2) + 36 # Distances of the camera in each calibration image
    files = sorted(os.listdir('calibrate')) # Get list of filenames for calibration images

    ptslist = []
    for i,file in enumerate(files):
        img = cv2.imread('calibrate/'+file)
        pts = getpts(img)
        uvw = sortpts(pts)
        ptslist.append(z[i]*uvw)

    uvw = np.vstack(ptslist).T

    d = 2.75 # Distance between adjacent corners of the checkerboard in cm

    ## Array of x and y positons of corners of the checkboard relative to the
    ## upper-left corner of the pattern since the origin of pixels is at the
    ## top-left of the image.
    xy = d*np.array([[(x,y) for y in range(0,8)] for x in range(0,10)]*11).reshape(880,2)

    zcol = (z*np.ones((80,1))).T.reshape(880,1) # Z column to point position array
    xyz = np.concatenate((xy,zcol),1).T # Add Z value to the array


    K = (uvw)@np.linalg.pinv(xyz) # Compute the intrinsic camera matrix

    return K,findRs(xyz.T[:80])

if __name__ == '__main__':
    calibrate_camera()
