import numpy as np
# from pylab import *
import matplotlib as f
# matplotlib.use('PS')   # generate postscript output by default
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.style as mplstyle
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib import rc
from cassinilib import DataPaths
from cassinilib.Vector import *
from cassinilib.Plot import *
from cassinilib.NewSignal import NewSignal
# mpl.rc('lines', linewidth=2, color='r')

def rotateAboutX(x0, y0, z0, theta):
    x = x0
    y = np.cos(theta)*y0 - np.sin(theta)*z0
    z = np.sin(theta)*y0 + np.cos(theta)*z0
    return (x,y,z)

def rotateAboutY(x0, y0, z0, theta):
    x = np.cos(theta)*x0 + np.sin(theta)*z0
    y = y0
    z = -np.sin(theta)*x0 + np.cos(theta)*z0
    return (x,y,z)

def ToMagneticCoords(R, B, dB):
    # NORMALIZE R & B VECTORS
    R_ = R/np.linalg.norm(R)
    B_ = B/np.linalg.norm(B)
    # PARALLEL DB COMPONENT TO B
    dB_par = np.dot(dB, B_) * B_
    # PERPENDICULAR COORD 1 IS DEFINED ALONG R X B
    dB_perp1_ = np.cross(R_, B_) / np.linalg.norm(np.cross(R_, B_))
    dB_perp1 = np.dot(dB, dB_perp1_) * dB_perp1_
    # THE REMAINING PERPENDICULAR COMPONENT
    dB_perp2_ = np.cross(B_, dB_perp1_)
    dB_perp2 = np.dot(dB, dB_perp2_) * dB_perp2_
    # return [np.linalg.norm(dB_par), np.linalg.norm(dB_perp1), np.linalg.norm(dB_perp2)]
    return [np.dot(dB, B_), np.dot(dB, dB_perp1_), np.dot(dB, dB_perp2_)]

def R_SPH2CAR(theta,phi):
    R = [[np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi),-np.sin(phi)],
         [np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi), np.cos(phi)],
         [np.cos(theta)            ,-np.sin(theta)            , np.zeros_like(theta)]]
    return np.array(R)
    # return R

def R_CAR2SPH(theta,phi):
    R = [[ np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi),  np.cos(theta)],
         [ np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)],
         [-np.sin(phi)              , np.cos(phi)              ,  np.zeros_like(theta)]]
    return np.array(R)
    # return R

def ToMagneticCoordsElement(R, B, BAVG, COORDS='KSM'):
    R = np.asarray(R)
    B = np.asarray(B)
    BAVG = np.asarray(BAVG)

    if COORDS == 'KRTP' or COORDS == 'RTN':
        B = np.dot(R_SPH2CAR(R[1], R[2]), B)
        PHI = R_SPH2CAR(R[1], R[2])[:,2]
        PHI_ = PHI / np.linalg.norm(PHI)
        BPAR_ = np.dot(R_SPH2CAR(R[1], R[2]), BAVG)
        BPAR_ = BPAR_ / np.linalg.norm(BPAR_)
        BPERP1_ = np.cross(PHI_,  BPAR_)   / np.linalg.norm(np.cross(PHI_,  BPAR_))
        BPERP2_ = np.cross(BPAR_, BPERP1_) / np.linalg.norm(np.cross(BPAR_, BPERP1_))
        return [np.dot(B, BPAR_),
                np.dot(B, BPERP1_),
                np.dot(B, BPERP2_)]

    if COORDS == 'KSO' or COORDS == 'KSM':
        r = np.linalg.norm(R)
        R = np.array([r, np.arccos(R[2] / r), np.arctan2(R[1], R[0])])
        PHI = R_CAR2SPH(R[1], R[2])[2,:]
        PHI_ = PHI / np.linalg.norm(PHI)
        BPAR_ = BAVG
        BPAR_ = BPAR_ / np.linalg.norm(BPAR_)
        BPERP1_ = np.cross(PHI_, BPAR_)    / np.linalg.norm(np.cross(PHI_, BPAR_))
        BPERP2_ = np.cross(BPAR_, BPERP1_) / np.linalg.norm(np.cross(BPAR_, BPERP1_))
        return [np.dot(B, BPAR_),
                np.dot(B, BPERP1_),
                np.dot(B, BPERP2_)]

def ToMagneticCoords2(R, B, BAVG, COORDS='KSM', diagnostic=False):
    R = np.asarray(R)
    B = np.asarray(B)
    BAVG = np.asarray(BAVG)

    if COORDS == 'KRTP' or COORDS == 'RTN':
        R = R.T
        PHI = R_CAR2SPH(R[:,1], R[:,2])[2,:].T
        PHI_ = PHI / np.linalg.norm(PHI, axis=1)
        B = np.einsum('...ik,kj...->j...', R_SPH2CAR(R.T[1], R.T[2]).T, B.T)

        B = np.dot(R_SPH2CAR(R[1], R[2]), B)
        BAVG = np.dot(R_SPH2CAR(R[1], R[2]), BAVG)
        PHI = R_CAR2SPH(R[:,1], R[:,2])[2,:].T
        PHI_ = PHI / np.linalg.norm(PHI, axis=1)

        # Convert to spherical coords
        r = np.linalg.norm(R.T, axis=1)
        R = np.array([r, np.arccos(R[2]/r), np.arctan2(R[1],R[0])]).T

        PHI = R_CAR2SPH(R[:,1], R[:,2])[2,:].T
        PHI_ = PHI / np.linalg.norm(PHI, axis=1)

        BPAR_ = BAVG.T / np.linalg.norm(BAVG.T, axis=1)
        BPERP1_ = np.cross(PHI_,  BPAR_)   / np.linalg.norm(np.cross(PHI_,  BPAR_),   axis=1)
        BPERP2_ = np.cross(BPAR_, BPERP1_) / np.linalg.norm(np.cross(BPAR_, BPERP1_), axis=1)
        BPAR   = np.einsum('ik,ik->i', B.T, BPAR_)
        BPERP1 = np.einsum('ik,ik->i', B.T, BPERP1_)
        BPERP2 = np.einsum('ik,ik->i', B.T, BPERP2_)
        return [BPAR, np.dot(BC, BPERP1_), np.dot(BC, BPERP2_)]

    if COORDS == 'KSO' or COORDS == 'KSM':
        # Convert to spherical coords
        r = np.linalg.norm(R.T, axis=1)
        R = np.array([r, np.arccos(R[2]/r), np.arctan2(R[1],R[0])]).T

        PHI = R_CAR2SPH(R[:,1], R[:,2])[2,:].T
        PHI_ = PHI / np.linalg.norm(PHI, axis=1)

        BPAR_ = BAVG.T / np.linalg.norm(BAVG.T, axis=1)
        BPERP1_ = np.cross(PHI_,  BPAR_)   / np.linalg.norm(np.cross(PHI_,  BPAR_),   axis=1)
        BPERP2_ = np.cross(BPAR_, BPERP1_) / np.linalg.norm(np.cross(BPAR_, BPERP1_), axis=1)
        BPAR   = np.einsum('ik,ik->i', B.T, BPAR_)
        BPERP1 = np.einsum('ik,ik->i', B.T, BPERP1_)
        BPERP2 = np.einsum('ik,ik->i', B.T, BPERP2_)

        # VERIFY ORTHOGONALITY
        # print(np.all(np.einsum('ik,ik->i',BPAR_, BPERP1_) == 0))
        # print(np.all(np.einsum('ik,ik->i',BPAR_, BPERP2_) == 0))
        # print(np.all(np.einsum('ik,ik->i',BPERP1_, BPERP2_) == 0))

        return [BPAR, BPERP1, BPERP2]

def fieldTransform(COORDS, FIELDS, IN='KSM'):
    ''' Transform the Field into Field-Aligned Coordinates '''
    # Input Vectors for Field Transformation
    R = [COORDS[0].y, COORDS[1].y, COORDS[2].y]
    B = [FIELDS[0].y, FIELDS[1].y, FIELDS[2].y]
    B_avg = [FIELDS[0].run_avg, FIELDS[1].run_avg, FIELDS[2].run_avg]

    BArray = []
    # Perform Transformations with Array Forecasting (works only for KSM)
    # BArray = ToMagneticCoords2(R, B, B_avg, COORDS=IN)

    # Perform Transformations one at a time (slower, but works well!)
    for i in range(len(COORDS[0].y)):
        R_ = [COORDS[0].y[i], COORDS[1].y[i], COORDS[2].y[i]]
        B_ = [FIELDS[0].y[i], FIELDS[1].y[i], FIELDS[2].y[i]]
        B_avg_ = [FIELDS[0].run_avg[i],
                  FIELDS[1].run_avg[i],
                  FIELDS[2].run_avg[i]]
        Bpar, Bperp1, Bperp2 = ToMagneticCoordsElement(R_, B_, B_avg_,
                                                       COORDS=IN)
        BArray.append([Bpar, Bperp1, Bperp2])

    # Transpose to get components more easily
    BArray = np.asarray(BArray).T

    N = COORDS[0].N
    dt = COORDS[0].dt
    Bpar = NewSignal(N=N, dt=dt, name='Bpar',
                     units='nT', kind='Field Perturbations',
                     data=BArray[0])
    Bperp1 = NewSignal(N=N, dt=dt, name='Bperp1',
                       units='nT', kind='Field Perturbations',
                       data=BArray[1])
    Bperp2 = NewSignal(N=N, dt=dt, name='Bperp2',
                       units='nT', kind='Field Perturbations',
                       data=BArray[2])
    Btot_mag = np.linalg.norm([Bpar.y, Bperp1.y, Bperp2.y], axis=0)
    Btot = NewSignal(N=N, dt=dt, name='Btot',
                     units='nT', kind='Field Perturbations',
                     data=Btot_mag)
    return [Bpar, Bperp1, Bperp2, Btot]

if __name__ == '__main__':
    FIELDS = []
    COORDS = []
    # fieldTransform(COORDS, FIELDS, IN='KSM', OUT='Blocal', diagnostic=False)


    # R = [COORDS[0].y, COORDS[1].y, COORDS[2].y]
    # B = [FIELDS[0].y, FIELDS[1].y, FIELDS[2].y]
    # dB = [FIELDS[0].dy, FIELDS[1].dy, FIELDS[2].dy]
    # B_avg = [FIELDS[0].run_avg, FIELDS[1].run_avg, FIELDS[2].run_avg]

    R = [10, 0, 0]
    B = [0.1, 0, 2]
    B_avg = [0, 0, 2]

    BArray = ToMagneticCoords2(R, B, B_avg, COORDS=IN, diagnostic=False)

    Bpar = BArray[0]
    Bperp1 = BArray[1]
    Bperp2 = BArray[2]


    print('Bpar = ', Bpar)
    print('Bperp1 = ', Bperp1)
    print('Bperp2 = ', Bperp2)
    # ToMagneticCoordsElement(R, B, BAVG, dB, COORDS='KSM', diagnostic=False):
