import datetime
import numpy as np
import matplotlib as f
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.style as mplstyle
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import proj3d
from matplotlib import rc
from cassinilib import DataPaths
from cassinilib.Vector import *
from cassinilib.ToMagneticCoords import *
from cassinilib.DataPaths import *
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
import copy
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
import matplotlib.ticker as mticker
from wavesolver.io import *
# from pylab import *
# from collections import OrderedDict
# from matplotlib import dates
# from matplotlib import rc
# import matplotlib.colors as colors
# import cassinilib.ToMagneticCoords
# from cassinilib import ToMagneticCoords

fancy_title = {
            "r" : r'$R$ ($R_S$)',
            "th" : r'$\theta$',
            "phi" : r'$\phi$',
            "Br" : r'$B_{r}$ (nT)',
            "Bth" : r'$B_{\theta}$ (nT)',
            "Bphi" : r'$B_{\phi}$ (nT)',
            "dBpar" : r'$dB_{\parallel}$ (nT)',
            "dBperp1": r'$dB_{\bot 1}$ (nT)',
            "dBperp2": r'$dB_{\bot 2}$ (nT)',
            "SIM 1": r'$SIM 1$ (nT)',
            "SIM 1. Noisy": r'$SIM 1. Noisy$ (nT)'
}
clist = ['limegreen', 'firebrick', 'slateblue', 'brown', 'chocolate', 'coral', 'hotpink', 'royalblue', 'mediumaquamarine', 'darkslategrey']
clist_boring = ['black', 'gray']
# Tlist = [10, 20, 30, 45, 60, 75, 90, 120, 180, 600]
Tlist = [20, 30, 45, 60, 75, 90, 120]


#10 RS
orbit10LT1 = [0.41, 1.24, 2.07, 2.9, 3.73, 4.56, 5.39, 6.22, 7.05, 7.88, 8.71,
              9.53, 10.36, 11.18, 12.01, 12.01, 12.83, 13.66, 14.48, 15.3,
              16.13, 16.96, 17.78, 18.61, 19.43, 20.26, 21.09, 21.92, 22.75,
              23.58]
orbit10LAT1 = [73.08, 73.11, 73.16, 73.09, 73.15, 73.21, 73.13, 73.12, 73.14,
               73.28, 73.27, 73.24, 73.21, 73.32, 73.3, 73.35, 73.34, 73.34,
               73.2, 73.21, 73.21, 73.2, 73.19, 73.2, 73.16, 73.13, 73.09,
               73.06, 73.19, 73.05]
orbit10LT2 = [0.42, 1.25, 2.08, 2.91, 3.74, 4.56, 5.39, 6.22, 7.04, 7.87,
              8.69, 9.52, 10.34, 11.17, 11.99, 11.99, 12.82, 13.64, 14.47,
              15.29, 16.12, 16.95, 17.78, 18.61, 19.44, 20.27, 21.1, 21.93,
              22.76, 23.59]
orbit10LAT2 = [-71.31, -71.25, -71.27, -71.26, -71.35, -71.4, -71.42, -71.33,
               -71.33, -71.48, -71.51, -71.43, -71.41, -71.46, -71.53, -71.44,
               -71.45, -71.43, -71.48, -71.43, -71.41, -71.4, -71.4, -71.33,
               -71.29, -71.36, -71.24, -71.36, -71.3, -71.33]

#20 RS
orbit20LT1 = [0.37, 1.21, 2.05, 2.9, 3.74, 4.59, 5.45, 6.3, 7.15, 7.99, 8.83,
              9.66, 10.47, 11.26, 12.04, 12.04, 12.82, 13.6, 14.4, 15.22,
              16.04, 16.87, 17.7, 18.54, 19.37, 20.2, 21.03, 21.87, 22.7,
              23.53]
orbit20LAT1 = [75.34, 75.37, 75.28, 75.41, 75.45, 75.59, 75.64, 75.83, 75.97,
               76.22, 76.42, 76.73, 76.87, 76.98, 77.03, 77.05, 77.06, 76.85,
               76.77, 76.45, 76.28, 76.1, 75.84, 75.69, 75.59, 75.55, 75.4,
               75.33, 75.38, 75.34]
orbit20LT2 = [0.46, 1.3, 2.13, 2.96, 3.8, 4.63, 5.46, 6.29, 7.13, 7.96, 8.78,
              9.6, 10.39, 11.18, 11.96, 11.96, 12.74, 13.53, 14.34, 15.17,
              16.01, 16.85, 17.7, 18.56, 19.41, 20.26, 21.1, 21.95, 22.79,
              23.63]
orbit20LAT2 = [-73.68, -73.76, -73.75, -73.87, -73.98, -74.04, -74.24, -74.37,
               -74.61, -74.8, -75.04, -75.29, -75.49, -75.58, -75.65, -75.63,
               -75.64, -75.44, -75.29, -75.05, -74.76, -74.47, -74.32, -74.17,
               -73.98, -73.91, -73.76, -73.71, -73.79, -73.78]

#25 RS
orbit25LT1 = [0.34, 1.19, 2.04, 2.9, 3.76, 4.63, 5.5, 6.39, 7.29, 8.19, 9.1,
              10.01, 10.87, 11.57, 12.1, 12.1, 12.61, 13.27, 14.09, 14.97,
              15.85, 16.74, 17.61, 18.47, 19.32, 20.16, 20.99, 21.83, 22.66,
              23.5]
orbit25LAT1 = [75.69, 75.71, 75.83, 75.95, 76.05, 76.17, 76.48, 76.76, 77.21,
               77.7, 78.33, 79.1, 79.91, 80.63, 80.65, 80.7, 80.64, 80.03,
               79.21, 78.46, 77.83, 77.29, 76.87, 76.55, 76.27, 76.09, 75.98,
               75.79, 75.79, 75.71]
orbit25LT2 = [0.5, 1.33, 2.17, 3.0, 3.84, 4.68, 5.53, 6.39, 7.26, 8.14, 9.03,
              9.91, 10.73, 11.38, 11.9, 11.9, 12.43, 13.13, 13.99, 14.9, 15.81,
              16.72, 17.61, 18.5, 19.37, 20.24, 21.1, 21.96, 22.81, 23.65]
orbit25LAT2 = [-74.17, -74.29, -74.28, -74.43, -74.62, -74.81, -75.06, -75.41,
               -75.92, -76.52, -77.21, -78.07, -78.92, -79.63, -79.71, -79.68,
               -79.56, -78.83, -77.91, -77.12, -76.42, -75.83, -75.34, -75.0,
               -74.74, -74.52, -74.39, -74.3, -74.25, -74.19]

# Enceladus
#KSM
orbitEnceladusLT1 =  [0.41, 1.24, 2.07, 2.9, 3.72, 4.55, 5.38, 6.21, 7.04,
                      7.86, 8.69, 9.52, 10.34, 11.18, 12.0, 12.0, 12.83,
                      13.65, 14.48, 15.31, 16.14, 16.96, 17.79, 18.62, 19.45,
                      20.27, 21.1, 21.93, 22.76, 23.58]
orbitEnceladusLAT1 =  [65.17, 65.18, 65.17, 65.15, 65.17, 65.16, 65.2, 65.05,
                       65.01, 65.01, 65.05, 65.04, 65.06, 65.05, 65.05, 65.05,
                       65.04, 65.0, 65.04, 65.05, 65.03, 65.04, 65.02, 65.14,
                       65.19, 65.15, 65.18, 65.15, 65.21, 65.16]
orbitEnceladusLT2 =  [0.42, 1.24, 2.07, 2.89, 3.73, 4.55, 5.38, 6.21, 7.03,
                      7.86, 8.69, 9.51, 10.35, 11.17, 12.0, 12.0, 12.83,
                      13.65, 14.48, 15.31, 16.14, 16.97, 17.79, 18.62, 19.45,
                      20.28, 21.11, 21.93, 22.76, 23.58]
orbitEnceladusLAT2 =  [-62.13, -62.12, -62.1, -62.09, -62.1, -62.07, -62.06,
                       -62.28, -62.24, -62.27, -62.22, -62.24, -62.25, -62.26,
                       -62.26, -62.26, -62.26, -62.25, -62.27, -62.27, -62.25,
                       -62.26, -62.25, -62.12, -62.12, -62.14, -62.09, -62.1,
                       -62.12, -62.13]
# orbitEnceladus
#RHEA
orbitRheaLT1 =  [0.4, 1.24, 2.06, 2.9, 3.73, 4.56, 5.38, 6.21, 7.04, 7.88,
                 8.69, 9.52, 10.35, 11.18, 12.0, 12.0, 12.83, 13.65, 14.48,
                 15.31, 16.14, 16.95, 17.79, 18.62, 19.44, 20.27, 21.1, 21.92,
                 22.74, 23.58]
orbitRheaLAT1 =  [72.33, 72.34, 72.34, 72.49, 72.45, 72.43, 72.49, 72.5,
                  72.51, 72.48, 72.52, 72.43, 72.52, 72.51, 72.44, 72.44,
                  72.61, 72.46, 72.51, 72.57, 72.49, 72.57, 72.5, 72.47,
                  72.39, 72.45, 72.37, 72.39, 72.47, 72.49]
orbitRheaLT2 =  [0.42, 1.25, 2.08, 2.91, 3.74, 4.56, 5.39, 6.21, 7.04, 7.86,
                 8.69, 9.52, 10.34, 11.16, 11.99, 11.99, 12.82, 13.65, 14.47,
                 15.3, 16.12, 16.96, 17.79, 18.61, 19.45, 20.27, 21.1, 21.93,
                 22.76, 23.59]
orbitRheaLAT2 =  [-70.38, -70.52, -70.46, -70.42, -70.49, -70.47, -70.53,
                  -70.54, -70.62, -70.51, -70.6, -70.55, -70.52, -70.67,
                  -70.49, -70.49, -70.51, -70.59, -70.52, -70.56, -70.54,
                  -70.46, -70.52, -70.5, -70.41, -70.5, -70.5, -70.37,
                  -70.41, -70.37]
#TITAN
#KSM
orbitTitanLT1 =  [0.76, 1.6, 2.44, 3.3, 4.15, 5.01, 5.86, 6.73, 7.58, 8.43,
                  9.26, 10.07, 10.87, 11.66, 12.42, 13.2, 13.99, 14.8, 15.62,
                  16.44, 17.28, 18.12, 18.95, 19.77, 20.6, 21.43, 22.27,
                  23.08, 23.92, 23.92]
orbitTitanLAT1 =  [75.41, 75.49, 75.58, 75.64, 75.69, 75.75, 75.97, 76.18,
                   76.38, 76.55, 76.82, 76.96, 77.13, 77.25, 77.24, 77.19,
                   77.0, 76.78, 76.61, 76.42, 76.23, 76.02, 75.9, 75.77,
                   75.6, 75.57, 75.53, 75.46, 75.42, 75.42]
orbitTitanLT2 =  [0.03, 0.03, 0.88, 1.71, 2.55, 3.39, 4.22, 5.05, 5.89, 6.73,
                  7.56, 8.39, 9.2, 10.01, 10.8, 11.56, 12.34, 13.11, 13.92,
                  14.73, 15.56, 16.4, 17.26, 18.11, 18.97, 19.81, 20.67,
                  21.52, 22.35, 23.2]
orbitTitanLAT2 =  [-73.94, -73.94, -73.92, -73.96, -73.91, -74.09, -74.15,
                   -74.3, -74.42, -74.59, -74.86, -75.17, -75.37, -75.68,
                   -75.83, -75.97, -75.86, -75.8, -75.6, -75.3, -75.11,
                   -74.82, -74.65, -74.43, -74.21, -74.13, -74.0, -73.96,
                   -73.91, -73.95]

def printMPLversion():
    print('Using Matplotlib version: ' + f.__version__)
    print('Install location: ' + f.__file__)
    print('mpl.get_configdir: ' + f.get_configdir())
    print('matplotlibrc file: ' + f.matplotlib_fname())

def loadMPLstyles():
    mplstyle.use(['ggplot'])
    f.rcParams['path.simplify_threshold'] = 1.0
    mplstyle.use('fast')

def set_saveStyle():
    # mpl.use('PS')   # generate postscript output by default
    rc('savefig', dpi=300)       # higher res outputs

def setPub3D(ax):
    rc('font', weight='light')    # bold fonts are easier to see
    # rc('lines', lw=1, color='k') # thicker black lines
    # rc('grid', c='0.5', ls='-', lw=0.5)  # solid gray grid lines
    # mpl.rcParams['text.color'] = 'w'
    # mpl.rcParams['xtick.color'] = 'red'
    # mpl.rcParams['ytick.color'] = 'red'
    # mpl.rcParams['axes.labelcolor'] = 'r'
    ax.set_aspect('equal')
    setAxesEqual(ax)
    # ax.w_xaxis.line.set_color("red")
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.set_facecolor('xkcd:salmon')
    ax.set_facecolor('white')
    # ax.w_xaxis.set_pane_color((R,G,B,A))
    # ax.w_yaxis.set_pane_color((R,G,B,A))
    # ax.w_zaxis.set_pane_color((R,G,B,A))
    # Get rid of colored axes planes
    # First remove fill
    # ax.xaxis.pane.fill = False
    # ax.yaxis.pane.fill = False
    # ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    # Bonus: To get rid of the grid as well:
    # ax.grid(False)
    # ax.Axes.grid(color='red', linestyle='-', linewidth=1, alpha=0.2, which='major', axis='both')
    # ax.grid(color='red', linestyle='-', linewidth=1, alpha=0.2)
    ax.set_proj_type('ortho')
    # ax.set_proj_type('persp')
    # ax.set_frame_on(False)
    # plt.tight_layout()
    # mplstyle.use(['ggplot'])

def setAxes3D(ax, origin=None, axisWidth=None, labels=['X','Y','Z'], title=None, axisLim=None, viewAngle=[30,0]):
    if axisLim: PlotAxes(ax, axlim=axisLim)
    if origin is None: origin = [0,0,0]
    if axisWidth is not None:
        ax.set_xlim3d(origin[0]-axisWidth[0]/2, origin[0]+axisWidth[0]/2)
        ax.set_ylim3d(origin[1]-axisWidth[1]/2, origin[1]+axisWidth[1]/2)
        ax.set_zlim3d(origin[2]-axisWidth[2]/2, origin[2]+axisWidth[2]/2)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.view_init(viewAngle[0], viewAngle[1])
    if title is not None: ax.set_title(title)

def drawSphere(ax, R, r, color="black", alpha=0.2, rotate=False, N=40):
    u, v = np.mgrid[0:2*np.pi:N*1j, 0:np.pi:N/2j]
    # u = np.array(u)
    # v = np.array(v)
    # print(u)
    # ar = ToMagneticCoords.R_SPH2CAR(v,u)
    # ar = np.asarray(ToMagneticCoords.R_SPH2CAR(v,u))[:,0]
    # print(ToMagneticCoords.R_SPH2CAR(v,u))
    # ar = ToMagneticCoords.R_SPH2CAR2(v,u)[:,0]
    # ar = np.array(ToMagneticCoords.R_SPH2CAR(v,u))
    ar = np.array(R_SPH2CAR(v,u))
    # print('ar = ', ar.shape)
    ar = ar[:,0]
    # print('ar = ', ar.shape)
    # ar = ar.transpose(2,0,1).reshape(3,-1)
    # ar = ar.reshape(3,-1)
    # ar = ar.reshape((ar.shape[0], -1))
    elem = ar[0]
    # ar=np.asarray([ar[0],ar[1],ar[2]])
    # print('ar = ', ar.shape)
    # print(R)
    # print(np.full_like(elem,R[0]))
    # r = np.asarray([np.full_like(elem,r[0]),np.full_like(elem,r[1]),np.full_like(elem,r[2])])
    R = np.asarray([np.full_like(elem,R[0]),np.full_like(elem,R[1]),np.full_like(elem,R[2])])
    # ar = ToMagneticCoords.R_SPH2CAR2(v,u)[:][0]

    # print(ar.shape)
    # ar = np.array(ToMagneticCoords.R_SPH2CAR2(v,u))[:][0]
    # ar = ar[:,0]
    # ar = ToMagneticCoords.R_SPH2CAR(v,u)[:,0]
    if rotate:
        theta_y = DataPaths.SATURN_AXISROT_Y
        theta_x = DataPaths.SATURN_AXISROT_X
        x = r*x + R[0]
        y = r*y + R[1]
        z = r*z + R[2]
        (x,y,z) = ToMagneticCoords.rotateAboutY(x,y,z,theta_y)
        (x,y,z) = ToMagneticCoords.rotateAboutX(x,y,z,theta_x)
    # SHIFT AND SCALE SPHERE
    RC = R + r*ar
    # RC = R + r*ar[:,0]
    # ax.plot_wireframe(RC[0], RC[1], RC[2], color=color, alpha=alpha, linewidth=2, zorder=1)
    surf = ax.plot_surface(RC[0], RC[1], RC[2], rcount=N, ccount=N/2,
                       color=color, shade=True, zorder=0)
    surf.set_facecolor((1,1,1,0.2))

def drawOrbit(ax, R=(0,0), r=4, color="#ffd000", alpha=0.8, rotate=False, linewidth=1.5, z=0):
    # Draw a circle on the x=0 'wall'
    p = Circle(R, r, linestyle='--', color=color, fill=None, linewidth=linewidth, alpha=alpha)
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=z, zdir="z")

def drawRing(ax, radius, width, color='black', alpha=0.2, rotate=False):
    theta = np.linspace(0, 2.*np.pi, 25)
    phi = np.linspace(0, 2*np.pi, 25)
    theta, phi = np.meshgrid(theta, phi)
    x = (radius + width*np.cos(theta)) * np.cos(phi)
    y = (radius + width*np.cos(theta)) * np.sin(phi)
    z = np.zeros((len(theta), len(theta)), dtype=float)
    if rotate:
        theta_y = DataPaths.SATURN_AXISROT_Y*np.pi/180.
        theta_x = DataPaths.SATURN_AXISROT_X*np.pi/180.
        (x,y,z) = ToMagneticCoords.rotateAboutY(x,y,z,theta_y)
        (x,y,z) = ToMagneticCoords.rotateAboutX(x,y,z,theta_x)
    ax.plot_wireframe(x, y, z, color=color, alpha=alpha)


def PlotAxes(ax, axlim=None, color='white', alpha=0.8, lw=2, mutation_scale=8):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    # ax.plot([xlim[0], xlim[1]], [0,0], [0,0], color="black", linestyle="-", lw=1, alpha=1)
    # ax.plot([0,0], [ylim[0], ylim[1]], [0,0], color="black", linestyle="-", lw=1, alpha=1)
    # ax.plot([0,0], [0,0], [zlim[0], zlim[1]], color="black", linestyle="-", lw=1, alpha=1)
    if axlim:
        x = Arrow3D([-axlim, axlim], [0, 0], [0, 0], mutation_scale=mutation_scale, lw=lw, arrowstyle="-|>", color=color, alpha=alpha)
        y = Arrow3D([0,0], [-axlim, axlim], [0, 0], mutation_scale=mutation_scale, lw=lw, arrowstyle="-|>", color=color, alpha=alpha)
        z = Arrow3D([0,0], [0,0], [-axlim, axlim], mutation_scale=mutation_scale, lw=lw, arrowstyle="-|>", color=color, alpha=alpha)
    else:
        x = Arrow3D([xlim[0], xlim[1]], [0, 0], [0, 0], mutation_scale=mutation_scale, lw=lw, arrowstyle="-|>", color=color, alpha=alpha)
        y = Arrow3D([0,0], [ylim[0], ylim[1]], [0, 0], mutation_scale=mutation_scale, lw=lw, arrowstyle="-|>", color=color, alpha=alpha)
        z = Arrow3D([0,0], [0,0], [zlim[0], zlim[1]], mutation_scale=mutation_scale, lw=lw, arrowstyle="-|>", color=color, alpha=alpha)
    ax.add_artist(x)
    ax.add_artist(y)
    ax.add_artist(z)

def projectPoints3D(ax, R, color="white", lw=1, alpha=0.7, radiusLine=False):
    dR0 = np.linspace(0, R[0], 2)
    dR1 = np.linspace(0, R[1], 2)
    dR2 = np.linspace(0, R[2], 2)
    N = len(dR0)
    ax.plot(dR0      ,[R[1]]*N ,[0]*N, color=color, linestyle="--", lw=lw, alpha=alpha)
    ax.plot([R[0]]*N   ,dR1     ,[0]*N, color=color, linestyle="--", lw=lw, alpha=alpha)
    if radiusLine:
        ax.plot([R[0], 0]   ,[R[1], 0],[0]*N, color=color, linestyle="--", lw=lw, alpha=alpha)
    ax.plot([R[0]]*N   ,[R[1]]*N  , dR2, color=color, linestyle="--", lw=lw, alpha=alpha)

def drawPlane(ax, R, N, xlim, ylim, color='black', alpha=0.1):
    point  = np.array(R)
    normal = np.array(N)
    # a plane is a*x+b*y+c*z+d=0
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    d = -point.dot(normal)
    # create x,y
    xx, yy = np.meshgrid(range(xlim[0],xlim[1]), range(ylim[0],ylim[1]))
    # calculate corresponding z
    z = (-normal[0] * xx - normal[1] * yy - d) * 0. /normal[2]
    # plot the surface
    ax.plot_surface(xx, yy, z, alpha=alpha, color = 'black', zorder=-1)

def addText3D(ax, R, text, c, zdir=0, alpha=1, box=True, fs=18):
    # label = '(%d, %d, %d), dir=%s' % (x, y, z, zdir)
    label = text
    color = c
    # ax.text(3, 8, 'boxed italics text in data coords', style='italic',
        # bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})

    # bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="cyan", ec="b", lw=2)
        # bbox_props = dict(boxstyle="square,pad=0.3", fc=color, ec="black", lw=1, alpha=0.5)
    # bbox_props = dict(facecolor=color, color=color, alpha=0.5,boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8))
    if box:
        bbox_props = dict(boxstyle="round,pad=0.25", fc=color, lw=1, alpha=0.5)
        ax.text(R[0], R[1], R[2], label, size=fs, rotation=0, color='white', ha="center", va="center",bbox=bbox_props, alpha=alpha)
    else:
        ax.text(R[0], R[1], R[2], label, size=fs, rotation=0, color=color, ha="center", va="center", alpha=alpha)
    # ax.text(x, y, z, label, size=7, rotation=-45., color=color, ha="center", va="center",bbox={'facecolor':'black', 'alpha':0.2, 'pad':2})
    # ax.text(x, y, z, label, size=18, color=color, ha="center", va="center",bbox={'facecolor':'black', 'alpha':0.2, 'pad':7})
    # ax.text(x, y, z, label, size=18, color=color, ha="center", va="center")

def addVector3D(ax, R, V, name=None, color='black', label=True, astyle="-|>", lw=2, lstyle='-', alpha=1, box=True, fs=18):
    P = Arrow3D([R[0], R[0]+V[0]],
                [R[1], R[1]+V[1]],
                [R[2], R[2]+V[2]],
                mutation_scale=10, lw=lw, arrowstyle=astyle, linestyle=lstyle, alpha=alpha,color=color)
    ax.add_artist(P)
    if label and name is not None:
        margin = abs(ax.get_xlim3d()[1]-ax.get_xlim3d()[0])*0.01
        V_ = V/np.linalg.norm(V) * margin
        addText3D(ax, R+V+V_, name, color, alpha, box=box, fs=fs)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def setAxesRadius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def setAxesEqual(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    limits = np.array([

        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    setAxesRadius(ax, origin, radius)

def plotVectorsOffCentre3D(R, RC, vectors, saveFig=False, FileName='3DVectors'):
    axisWidth = 5
    axisLim = int(round(R[0]*1.2))
    labels = [r'X $(R_S)$', r'Y $(R_S)$', r'Z $(R_S)$']
    Title = r'R: $\textbf{{{:3.1f}}}$ ($R_S$), LAT: $\textbf{{{:3.1f}}}$, PHI: $\textbf{{{:3.1f}}}$'.format(R[0],np.rad2deg(R[1]),np.rad2deg(R[2]))
    #================= PLOT ===================
    # mplstyle.use(['ggplot'])
    loadMPLstyles()
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    setPub3D(ax)
    drawSaturn3D(ax, drawPlane=False, drawRings=True)
    visualizePointFromOrigin(ax, RC, radialVector=True, projectPoint=True, drawImSphere=True, s=20, color='xkcd:salmon', label=r'$\hat{r}$')
    drawVectorList3D(ax, vectors)
    setAxes3D(ax, origin=RC, axisWidth=[axisWidth]*3, labels=labels, title=Title, axisLim=axisLim, viewAngle=[45,-90])
    if saveFig:
        plt.savefig(DataPaths.EXPORT_FOLDER + '/' + Filename + '.png', dpi = 200)
    else:
        plt.show()

def new3DAxes(axisWidth=20, viewAngle=[45,-90]):
    labels = [r'X $(R_S)$', r'Y $(R_S)$', r'Z $(R_S)$']
    loadMPLstyles()
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    setPub3D(ax)
    drawSaturn3D(ax, drawPlane=False, drawRings=True)
    setAxes3D(ax, labels=labels, axisWidth=[axisWidth]*3, viewAngle=viewAngle)
    return ax

def plotVectorsFromPoint3D(ax, RC, vectors, ptcolor='black'):
    visualizePointFromOrigin(ax, RC, radialVector=True, projectPoint=False, drawImSphere=False, s=10, color=ptcolor)
    drawVectorList3D(ax, vectors)

def visualizePointFromOrigin(ax, RC, radialVector=True, projectPoint=False, drawImSphere=False, s=20, color='r', alpha=1, label=None):
    ax.scatter(RC[0], RC[1], RC[2], s=s, c=color, alpha=alpha)
    if drawImSphere: drawSphere(ax, [0,0,0], np.linalg.norm(RC), 'xkcd:light grey', 0.5)
    if projectPoint: projectPoints3D(ax, RC, color)
    if radialVector:
        addVector3D(ax, [0,0,0], RC, label=False, color='black', lw=2, astyle="-", lstyle="--", box=False)
        # print(RC*0.5)
        addVector3D(ax, RC, RC*0.5, name=label, label=True, box=False, color='black', lw=2, astyle="-", lstyle="--", fs=18)

def drawVectorList3D(ax, vectors):
    for vector in vectors:
        addVector3D(ax, vector.R, vector.V, name=vector.name, color=vector.color, lw=2, fs=10)


def drawSaturn3D(ax, draw_Plane=False, draw_Rings=True):
    axisLim=5
    drawSphere(ax, [0,0,0], 1, 'black', 0.2)
    if draw_Rings:
        drawRing(ax, radius=2.3, width=0.7, color="black", alpha=0.2)
    if draw_Plane:
        drawPlane(ax, [0,0,0], [0,0,1], [-axisLim,1], [-axisLim, axisLim], color='white', alpha=0.5)
        drawPlane(ax, [0,0,0], [0,0,1], [0,axisLim], [-axisLim, axisLim], color='white', alpha=0.2)


def eventTitle(event):
    TITLE = ''
    if event.synthetic:
        TITLE = 'SIMULATION DATA'
    else:
        DateStr = event.datefrom.strftime('%Y-%m-%dT%H:%M:%S')
        DateStrShort = event.datefrom.strftime('%Y-%m-%d %H:%M')
        DayInYear = event.datefrom.strftime('%j')
        TITLE = DateStrShort + ' (day ' + str(int(DayInYear)) + ')'
        if event.LT is not None:
            TITLE += (', ' + str(round(event.LT,1)) + ' LT')
        if event.COORD is not None:
            TITLE += ', ' + str(round(event.COORD[1],1)) + r' $^{\circ}$ lat, ' + str(round(event.COORD[0],1)) + r' $R_S$'
    return TITLE


def add_vline(ax, x, color, text):
    ax.axvline(x=x, ls='--', linewidth=1, color=color, alpha = 0.5)
    ax.text(x, ax.get_ylim()[1], text, ha="center", va="bottom", fontsize=10, color=color, rotation=90)


def plot_periods(ax, T_list, clist=['black']):
    c_ind = 0
    for T in T_list:
        if c_ind >= len(clist): c_ind = 0
        color = clist[c_ind]
        if T >= 120:
            h = T // 60
            m = T % 60
            if m == 0.:
                text = str(int(h)) + ' h'
            else:
                text = str(int(h)) + ' h ' + str(int(round(m))) + ' min'
        else:
            text = str(int(round(T))) + ' min'
        add_vline(ax, 1./T, color, text)
        c_ind += 1

def processMovie2(filename, res='5136x2880', fps=30, output='movie.mp4', config=None):
    import os
    # if path is not None: fullpath = path + "/" + self.filename 
    # if path is None: fullpath = ioconfig.path + ioconfig.sep + filename
    # if ioconfig:
        # path = os.path.join(config.path, filename)
    # fullpath = './Output/' + 'orbit'
    # os.system("ffmpeg -framerate "+str(fps)+ " -i " + fullpath + "_%03d.png -vcodec mpeg4 -s:v " + res + " -y " + output)
    os.system("ffmpeg -framerate "+ str(fps) + " -i " + filename + " -vcodec mpeg4 -s:v " + res + " -y " + output)
    # os.system("ffmpeg -framerate 15 -i Output/solution_%03d.png -vcodec mpeg4 -s:v 5136x2880 -y movie.mp4")

def scatter3D(DATA_ROWS, coords='CAR', show=True, filename='3Dscatter.png'):
    fig = plt.figure(figsize=(10, 10))
    plt.style.use('dark_background')
    ax1 = fig.add_subplot(111, projection='3d')
    DATA_ROWS = np.array(DATA_ROWS)
    if coords == 'SPH':
        R = cassinilib.sph2car(R)
    elif coords == 'CAR':
        R = [DATA_ROWS[:,1], DATA_ROWS[:,2], DATA_ROWS[:,3]]
    plot = ax1.scatter(R[0], R[1], R[2], antialiased=False)
    # ax1.plot_surface(DX[0], DX[1], DX[2])
    ax1.set_xlabel(r'X ($R_S$)', color='#ffffff', size=13)
    ax1.set_ylabel(r'Y ($R_S$)', color='#ffffff', size=13)
    ax1.set_zlabel(r'Z ($R_S$)', color='#ffffff', size=13)
    fig.suptitle("STARTING POSITIONS", y=0.95, fontsize=18)
    ax1.set_xlim(-10, 10.)
    ax1.set_ylim(-10., 10.)
    ax1.set_zlim(-10, 10.)
    ax1.set_aspect('equal')
    ax1.grid(True, ls='--', color='#000000', alpha=0.5)
    ax1.w_xaxis.set_pane_color((0.43, 0.45, 0.48, 0.4))
    ax1.w_yaxis.set_pane_color((0.43, 0.45, 0.48, 0.4))
    ax1.w_zaxis.set_pane_color((0.43, 0.45, 0.48, 0.4))
    ax1.view_init(elev=30, azim=360)
    if show == True:
        plt.show()
    else:
        plt.savefig(filename, dpi = 200)

def plotOrbit3D(Data=None, fls=None, n=0, n_total=1, viewAngleChange=1, axisLim = 8, config=None):
    fig = plt.figure(figsize=(9, 9))
    plt.style.use('dark_background')
    # plt.rcParams['grid.color'] = "deeppink"
    # plt.rcParams['lines.linewidth'] = 4
    # plt.gca().patch.set_facecolor('white')
    ax1 = fig.add_subplot(111, projection='3d')
    # ax = fig.gca(projection='3d',  axisbg='gray')
    # plot = ax1.scatter(R[:,0], R[:,1], R[:,2], antialiased=False)
    alphas = np.linspace(0.05, 0.2, size(Data[0]))
    rgba_colors = np.zeros((size(Data[0]),4))
    # for red the first column needs to be one
    rgba_colors[:,0] = 0.5
    rgba_colors[:,1] = 0.9
    rgba_colors[:,2] = 0.5
    # the fourth column needs to be your alphas
    rgba_colors[:, 3] = alphas
    plot = ax1.scatter(Data[0], Data[1], Data[2], c=rgba_colors, antialiased=False, s=5, marker='o')
    ax1.plot(Data[0][-1], Data[1][-1], Data[2][-1], c='#ffd000', ms=7, marker='o')
    ax1.set_xlabel(r'X ($%s$)' % coords[0].units, color='#ffffff', size=13)
    ax1.set_ylabel(r'Y ($%s$)' % coords[0].units, color='#ffffff', size=13)
    ax1.set_zlabel(r'Z ($%s$)' % coords[0].units, color='#ffffff', size=13)
    ax1.set_title('Time: ' + str(CurrentTime), size=14)
    ax1.set_xlim(-axisLim, axisLim)
    ax1.set_ylim(-axisLim, axisLim)
    ax1.set_zlim(-axisLim, axisLim)
    fig.set_facecolor('#171717')
    ax1.set_facecolor('#171717')
    # ax1.grid(False) 
    ax1.w_xaxis.pane.fill = False
    ax1.w_yaxis.pane.fill = False
    ax1.w_zaxis.pane.fill = False
    ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.05))
    ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.05))
    ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.05))
    ax1.xaxis._axinfo["grid"].update({"linewidth":0.5,'color':'#383838', 'alpha':0.4})
    ax1.yaxis._axinfo["grid"].update({"linewidth":0.5,'color':'#383838', 'alpha':0.4})
    ax1.zaxis._axinfo["grid"].update({"linewidth":0.5,'color':'#383838', 'alpha':0.4})
    # ax1.set_aspect('auto')
    ax1.set_box_aspect((1, 1, 1))
    drawSphere(ax1, [0,0,0], 1, color='#f2d1a7', alpha=0.7)
    drawOrbit(ax1, R=(0,0), r=4, color="#ffd000", alpha=0.8, rotate=False, linewidth=1.5)
    drawOrbit(ax1, R=(0,0), r=1, color="#ffffff", alpha=0.2, rotate=False, linewidth=1)
    RefColatitude = 65
    drawOrbit(ax1, R=(0,0), r=1*np.cos(RefColatitude * np.pi / 180.), color="#ffd000", alpha=0.3, rotate=False, linewidth=1, z=1*np.sin(RefColatitude * np.pi / 180.))
    drawOrbit(ax1, R=(0,0), r=1*np.cos(RefColatitude * np.pi / 180.), color="#ffd000", alpha=0.3, rotate=False, linewidth=1, z=-1*np.sin(RefColatitude * np.pi / 180.))

    drawPlane(ax1, [0,0,0], [0,0,1], [-axisLim,1], [-axisLim, axisLim], color='black', alpha=0.2)
    drawPlane(ax1, [0,0,0], [0,0,1], [0,axisLim], [-axisLim, axisLim], color='black', alpha=0.05)
    addText3D(ax1, (axisLim-2, 1, 1), 'Sun', 'white', 0.7, box=True, fs=12)

    ax1.view_init(elev=30, azim=290 - n * viewAngleChange)    # an_total=1, x1.set_proj_type('ortho')
    ax1.set_proj_type('persp')

    PlotAxes(ax1, axlim=None, color='white', alpha=0.6, lw=2, mutation_scale=12)
    projectPoints3D(ax1, Data[:,-1], color="white", lw=1, alpha=0.5, radiusLine=False)

    for fl in fls:
        ax1.plot(fl.traceXYZ[:,0], fl.traceXYZ[:,1], fl.traceXYZ[:,2],
                ls='--', lw=1, c='white', zorder = 4)
        ax1.scatter(fl.traceXYZ[0,0], fl.traceXYZ[0,1], fl.traceXYZ[0,2],
                s=10, alpha=0.8, marker="o", c="yellow", zorder=3) #c='#00cc8f'
        ax1.scatter(fl.traceXYZ[-1,0], fl.traceXYZ[-1,1], fl.traceXYZ[-1,2],
                s=10, alpha=0.8, marker="o", c="#00cc8f", zorder=3)

    if config:
        if config.save:
            path = os.path.join(ioconfig.path, 'orbit' + '_%03d' % ioconfig.id + ioconfig.format)
            plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close()
        else:
            plt.show()
    else:
        plt.show()

def sortLists(list1, list2):
    zipped_lists = zip(list1, list2)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    list1, list2 = [ list(tuple) for tuple in  tuples]
    return list1, list2

def phi2LT(phi):
    return (phi / np.pi) * 12 + 12

def LT2phi(LT):
    LT = np.asarray(LT)
    return (LT) / 12 * np.pi

def plotMoonOrbits(ax,
                   moons=['Enceladus', 'Rhea', 'Titan'],
                   hemisphere='North',
                   polar=False):
    cl1 = '#84ff3d'
    cl2 = '#ffec3d'
    cl3 = '#ffb53d'

    if 'Titan' in moons:
        if hemisphere == 'North':
            x = LT2phi(orbitTitanLT1) if polar else orbitTitanLT1
            ax.plot(x, orbitTitanLAT1,
                    label=r'Titan (20.27 $R_S$)',
                    color='yellow', ls='--', lw=1.5, alpha=0.7)
        else:
            x = LT2phi(orbitTitanLT2) if polar else orbitTitanLT2
            ax.plot(x, orbitTitanLAT2,
                    label=r'Titan (20.27 $R_S$)',
                    color='yellow', ls='--', lw=1.5, alpha=0.7)
    if 'Rhea' in moons:
        if hemisphere == 'North':
            x = LT2phi(orbitRheaLT1) if polar else orbitRheaLT1
            ax.plot(x, orbitRheaLAT1,
                    label=r'Rhea (8.75 $R_S$)',
                    color='white', ls='--', lw=1.5, alpha=0.4)
        else:
            x = LT2phi(orbitRheaLT2) if polar else orbitRheaLT2
            ax.plot(x, orbitRheaLAT2,
                    label=r'Rhea (8.75 $R_S$)',
                    color='white', ls='--', lw=1.5, alpha=0.4)
    if 'Enceladus' in moons:
        if hemisphere == 'North':
            x = LT2phi(orbitEnceladusLT1) if polar else orbitEnceladusLT1
            ax.plot(x, orbitEnceladusLAT1,
                    label=r'Enceladus (3.94 $R_S$)',
                    color='#00d48a', ls='--', lw=1.5, alpha=0.7)
        else:
            x = LT2phi(orbitEnceladusLT2) if polar else orbitEnceladusLT2
            ax.plot(x, orbitEnceladusLAT2,
                    label=r'Enceladus (3.94 $R_S$)',
                    color='#00d48a', ls='--', lw=1.5, alpha=0.7)
    if '10' in moons:
        if hemisphere == 'North':
            x = LT2phi(orbit10LT1) if polar else orbit10LT1
            ax.plot(x, orbit10LAT1,
                    label=r'10 $R_S$',
                    color=cl1, ls='-', lw=2.5, alpha=0.3)
        else:
            x = LT2phi(orbit10LT2) if polar else orbit10LT2
            ax.plot(x, orbit10LAT2,
                    label=r'10 $R_S$',
                    color=cl1, ls='-', lw=2.5, alpha=0.3)
    if '20' in moons:
        if hemisphere == 'North':
            x = LT2phi(orbit20LT1) if polar else orbit20LT1
            ax.plot(x, orbit20LAT1,
                    label=r'20 $R_S$',
                    color=cl2, ls='--', lw=2.5, alpha=0.5)
        else:
            x = LT2phi(orbit20LT2) if polar else orbit20LT2
            ax.plot(x, orbit20LAT2,
                    label=r'20 $R_S$',
                    color=cl2, ls='--', lw=2.5, alpha=0.5)
    if '25' in moons:
        if hemisphere == 'North':
            x = LT2phi(orbit25LT1) if polar else orbit25LT1
            ax.plot(x, orbit25LAT1,
                    label=r'25 $R_S$',
                    color=cl3, ls=':', lw=2.5, alpha=0.5)
        else:
            x = LT2phi(orbit25LT2) if polar else orbit25LT2
            ax.plot(x, orbit25LAT2,
                    label=r'25 $R_S$',
                    color=cl3, ls=':', lw=2.5, alpha=0.5)


def update_ticks(x, pos):
    if x == -2:
        return '22'
    elif x == 26:
        return '2'
    # elif pos == 6:
        # return 'pos is 6'
    else:
        return x


def plotHeatmapAxis(ax,
                    data,
                    bins = None,
                    xRange = [0,24],
                    yRange = [-9,9],
                    tickPos = 'bottom',
                    maxValue = None,
                    minValue = None,
                    ylabel = '',
                    xlabel = 'Local Time',
                    moons = ['Enceladus', 'Rhea', 'Titan']):
    # moons = None
    # Copy cmap so that we can modify the colors (set_under)
    cmap = copy.copy(plt.cm.get_cmap("plasma")) # jet | OrRd
    cmap.set_under(color='black')
    cmap.set_bad(color='pink')
    cmap.set_over(color='grey')
#    cmap.set_extremes(bad='green', under='grey', over='white')
    if minValue is None:
        minValue = 0.
    if maxValue is None:
        maxValue = np.max(data)
    if bins is not None:
        data = data[bins[0] : bins[1], : ]
    mat = ax.matshow(data,
                     cmap = cmap,
                     vmin = minValue,
                     vmax = maxValue,
                     extent =[xRange[0],
                              xRange[1],
                              yRange[0],
                              yRange[1]],
                     interpolation ='nearest',
                     aspect='auto',
                     origin ='lower')
    if tickPos == 'top':
        if moons is not None:
            plotMoonOrbits(ax, moons=moons,
                           hemisphere='North')
        ax.tick_params(right=False, bottom=False, top=True,
                       labelright=False, labelbottom=False,
                       labeltop=False,rotation=0)
    else:
        if moons is not None:
            plotMoonOrbits(ax, moons=moons,
                           hemisphere='South')
        ax.tick_params(right=False, bottom=True, top=False,
                       labelright=False, labelbottom=True,
                       labeltop=False,rotation=0)
        ax.set_xlabel(xlabel, size=12)
        #  leg = ax.legend(loc='lower right', frameon=False, fontsize=11)
#        plt.setp(leg.get_texts(), color='#828282')
        #  plt.setp(leg.get_texts(), color='#fafafa')
    ax.axvline(x = 12, color='white', ls='--', alpha=0.6, lw=1)
    ax.set_ylabel(ylabel, size=12, color='#b3b3b3')
    ax.grid(color='white', linestyle='--', linewidth=0.8, alpha=0.25)
    ax.xaxis.set_major_locator(MultipleLocator(2))
#    ax.xaxis.set_major_locator(MultipleLocator(10))
#    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.set_facecolor('#171717')
    return mat


def plotHeatmap(matrix,
                xRange = [0, 24],
                yRange = [-90, 90],
                splitRange = None,
                splitBins = None,
                config = None,
                margin = None,
                suptitle = 'Cassini Cumulative Dwell Time along ' +
                           'Different Flux Tubes\n',
                colorRange = None,
                xlabel = 'Local Time',
                ylabel = r'Latitude ($^\circ$)',
                moons = None,
                cbar_label = 'Cumulative Dwell Time (Days)',
                matrix0 = None,
                matrix1 = None,
                minBound = 10,
                showHistogram = True,
                histMatrices = None,
                ):
    """ Plot a 2D np.array as a heatmap """

    #  colorRange = config.plot.cmap_range
    plotData = 'days'
    # Configure the Figure

    styleGlobal(font_size=12, config=config)

    (scale, vmin, vmax, vmin2, vmax2,
     norm, ticks, ticklabels, barlabel,
     cmap) = dataDisplayConfig(plotData)
    vmax_hist = vmax * 20

#    fig = plt.figure(figsize=(10,5))
    fig = plt.figure(figsize=(6,4), dpi=150)

    if colorRange is None:
        if matrix0 is None:
            minValue = 0
            maxValue = np.max(matrix)
        else:
            minValue = 0
            maxValue = np.max(matrix0)
        colorRange = [minValue, maxValue]

    if margin is not None:
        rangeIncrease = int((xRange[1]-xRange[0]) * margin / matrix.shape[1])
        xRange = [xRange[0] - rangeIncrease, xRange[1] + rangeIncrease]

    margin = margin[0] if isinstance(margin, np.ndarray) or isinstance(margin, list) else margin
    Lmatrix = matrix[:,0:margin]
    Rmatrix = matrix[:,-margin:]
    matrixExt = np.concatenate((Rmatrix, matrix), axis=1)
    matrixExt = np.concatenate((matrixExt, Lmatrix), axis=1)
    matrix = matrixExt

    left, bottom, width, height = (-rangeIncrease,
                                    yRange[0],
                                    rangeIncrease,
                                    yRange[1]-yRange[0])
    left2 = xRange[1] - rangeIncrease

    if showHistogram:
        left, width = 0.1, 0.7
        bottom, height = 0.1, 0.33
        spacing = 0.03

        rect_scatter = [left, bottom, width, height]
        rect_scatter2 = [left, bottom+height+spacing, width, height]
        rect_histx = [left, bottom + 2*height + 2*spacing, width, 0.15]
        rect_histy = [left + width + spacing, bottom, 0.08, height]
        rect_histy2 = [left + width + spacing, bottom+height+spacing, 0.08, height]

        ax = fig.add_axes(rect_scatter)
        ax2 = fig.add_axes(rect_scatter2)
        ax_histx = fig.add_axes(rect_histx, sharex=ax)
        ax_histy = fig.add_axes(rect_histy, sharey=ax)
        ax_histy2 = fig.add_axes(rect_histy2, sharey=ax2)

        if histMatrices is None:
            matrixHist = matrix[splitBins[1][0]:splitBins[1][1]]
            matrixHist2 = matrix[splitBins[0][0]:splitBins[0][1]]

            histx1 = np.sum(matrixHist, axis=0)
            histx2 = np.sum(matrixHist2, axis=0)
            histx = (histx1 + histx2)

            histy = np.sum(matrixHist, axis=1)
            histy2 = np.sum(matrixHist2, axis=1)
        else:
            histx = histMatrices[0]
            histy = histMatrices[1]
            histy2 = histMatrices[2]
#        print(xlabel)
#        print(ylabel)
        caxes1 = plotHeatmapAxis(ax2, matrix, bins = splitBins[0],
                                 xRange = xRange,
                                 yRange = splitRange[0],
                                 tickPos = 'top',
                                 moons = moons,
                                 minValue = vmin,
                                 maxValue = vmax,
                                 xlabel = xlabel,
                                 ylabel = ylabel,
                                 )
        caxes2 = plotHeatmapAxis(ax, matrix, bins = splitBins[1],
                                 xRange = xRange,
                                 yRange = splitRange[1],
                                 tickPos = 'bottom',
                                 moons = moons,
                                 minValue = vmin,
                                 maxValue = vmax,
                                 xlabel = xlabel,
                                 ylabel = ylabel,
                                 )
#        ax_histy.set_ylim(splitRange[1][0], splitRange[1][1])
#        ax_histy2.set_ylim(splitRange[0][0], splitRange[0][1])
#        ax_histy.plot(np.linspace(splitRange[0][0], splitRange[0][1], histx.size), histx)
#        ax_histx.plot(np.linspace(xRange[0], xRange[1], histx.size), histx)
        #  ax_histy.plot(histy, np.linspace(splitRange[1][0], splitRange[1][1], histy.size))
        #  ax_histy2.plot(histy2, np.linspace(splitRange[0][0], splitRange[0][1], histy2.size))
      #  ax_histx.hist(histx, np.linspace(xRange[0], xRange[1], histx.size),
        ax_histx.bar(np.linspace(xRange[0], xRange[1], histx.size), histx,
                     width=0.51,
                     align='edge',
                     linewidth=0.1,
                     edgecolor='yellow',
                     facecolor='#221d9f',
                     alpha=0.9)
        ax_histy.barh(np.linspace(splitRange[1][0], splitRange[1][1], histy.size), histy,
                     height=6,
                     align='edge',
                     linewidth=0,
                     edgecolor='yellow',
                     facecolor='#221d9f',
                     alpha=0.9)
        ax_histy2.barh(np.linspace(splitRange[0][0], splitRange[0][1], histy2.size), histy2,
                     height=6,
                     align='edge',
                     linewidth=0,
                     edgecolor='yellow',
                     facecolor='#221d9f',
                     alpha=0.9)
        ax_histy.tick_params(axis="y", labelleft=False)
        ax_histy.tick_params(axis="x", labelbottom=True)
        ax_histy2.tick_params(axis="y", labelleft=False)
        ax_histy2.tick_params(axis="x", labelbottom=False)
        ax_histy.set_ylim(splitRange[1][0], splitRange[1][1])
        ax_histy2.set_ylim(splitRange[0][0], splitRange[0][1])
        ax_histy.set_xlim(0, vmax_hist)
        ax_histy2.set_xlim(0, vmax_hist)
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histx.set_ylim(0, vmax_hist)
        ax_histx.set_xlim(-2, 26)
#        for label in ax_histy.get_xticklabels():
#            label.set_rotation(90)
        caxes = caxes1
        ax1 = ax
    elif splitBins:
        ax1 = fig.add_subplot(211)
        caxes1 = plotHeatmapAxis(ax1, matrix, bins = splitBins[0],
                                 xRange = xRange,
                                 yRange = splitRange[0],
                                 tickPos = 'top',
                                 moons = moons,
                                 minValue = vmin,
                                 maxValue = vmax,
                                 ylabel = ylabel,
                                 )
        rectL = plt.Rectangle((left, bottom), width, height,
                         facecolor="white", alpha=0.1)
        rectR = plt.Rectangle((left2, bottom), width, height,
                         facecolor="white", alpha=0.1)
        ax1.add_patch(rectL)
        ax1.add_patch(rectR)
        ax2 = fig.add_subplot(212)
        caxes2 = plotHeatmapAxis(ax2, matrix, bins = splitBins[1],
                                 xRange = xRange,
                                 yRange = splitRange[1],
                                 tickPos = 'bottom',
                                 moons = moons,
                                 minValue = vmin,
                                 maxValue = vmax,
                                 ylabel = ylabel,
                                 )
        rectL = plt.Rectangle((left, bottom), width, height,
                         facecolor="white", alpha=0.1)
        rectR = plt.Rectangle((left2, bottom), width, height,
                         facecolor="white", alpha=0.1)
        ax2.add_patch(rectL)
        ax2.add_patch(rectR)
        caxes = caxes1

    else:
        ax = fig.add_subplot(111)
        caxes = plotHeatmapAxis(ax, matrix, bins = None,
                                            xRange = xRange,
                                            yRange = yRange,
                                            tickPos = 'top',
                                            moons = moons,
                                            ylabel = ylabel,
                                )

    #  fig.subplots_adjust(right=0.8)
#    cbar_ax = fig.add_axes([0.81, 0.1, 0.03, 0.78])
    cbar_ax = fig.add_axes([0.88, 0.1, 0.03, 0.72])
    cbar = fig.colorbar(caxes, cax=cbar_ax)
    cbar.set_label(label = cbar_label,
                   size = 13,
                   color = '#b3b3b3')
    #  fig.suptitle(suptitle, size=18, color='#787878')
    fig.subplots_adjust(wspace=0.15, hspace=0.15)
#    fig.set_facecolor('#171717')
#    fig.set_facecolor('#1c1c1c')
    fig.set_facecolor('#1e1e1e')

    # labels = [item.get_text() for item in ax2.get_xticklabels()]
    # print(labels)
    # labels[0] = '24'
    # ax2.set_xticklabels(labels)
    # axes.xaxis.label.set_size(20)
    # [t.set_color('red') for t in ax1.xaxis.get_ticklines()]
    # [t.set_color('red') for t in ax1.xaxis.get_ticklabels()]
    # ax2.get_xticklabels()[1].set_label("grey")
    # labels = [item.get_text() for item in ax2.get_xticklabels()]

#    ax1.get_xticklabels()[1].set_color("grey")
#    ax1.get_xticklabels()[-2].set_color("grey")
#    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))

    # labels = ax1.get_xticks().tolist()
    # labels[1] = '22'
    # labels[-2] = '2'
    # ax1.set_xticklabels(labels)
#    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))
#    ax2.get_xticklabels()[1].set_color("grey")
#    ax2.get_xticklabels()[-2].set_color("grey")
    # labels = ax2.get_xticks().tolist()
    # labels[1] = '22'
    # labels[-2] = '2'
    # ax2.set_xticklabels(labels)

    #  plt.show()
    plt.savefig('Output/figure.png', facecolor=fig.get_facecolor(), edgecolor='none')
    #  if config:
        #  if config.save:
            #  path = os.path.join(config.plot.folder, 'heatmap')
            #  #  path = os.path.join('Output', 'heatmap' + '_%04d' % config.id +
                                #  #  config.format)
            #  plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor='none')
            #  plt.close()
        #  else:
            #  plt.show()
    #  else:
        #  plt.show()

def plotHeatmapPolarAxis(ax,
                    data,
                    bins = None,
                    xRange = [0,24],
                    yRange = [-90,90],
                    tickPos = 'bottom',
                    maxValue = None,
                    ylabel = 'Conjugate Latitude (degrees)',
                    xlabel = 'Local Time',
                    moons = ['Enceladus', 'Rhea', 'Titan']):
    # Copy cmap so that we can modify the colors (set_under)
    cmap = copy.copy(plt.cm.get_cmap("plasma")) # jet | OrRd
    cmap.set_under(color='black')
    if maxValue is None:
        maxValue = np.max(data)

    if bins is not None:
        data = data[bins[0] : bins[1], : ]
    ltbins = len(data[0,:])

    # lat = np.linspace(-90, -60, num=LATcut2)
    # lat = np.linspace(90, 60, num=LATcut2)
    # ax1.set_rlim(-90, -60)
    # ax1.set_rlim(90, 60)


    # lat = np.linspace(0, 30, num=(bins[1]-bins[0]))
    lt = np.linspace(0, 2*np.pi, num=ltbins)
    # lt = np.linspace(0, 24, num=ltbins)
    # ax.set_philim(0, 24)
    # [-90, 90]

    if tickPos == 'top':
        ax.set_rlim(90, 60)
        lat = np.linspace(90, 60, num=(bins[1]-bins[0]))
        plotMoonOrbits(ax, moons=moons, hemisphere='North', polar=True)
        data = np.flip(data, axis=0)
    if tickPos == 'bottom':
        ax.set_rlim(-90, -60)
        lat = np.linspace(-90, -60, num=(bins[1]-bins[0]))
        plotMoonOrbits(ax, moons=moons, hemisphere='South', polar=True)
    # mat = ax.pcolormesh(lt, lat, data, shading='auto', cmap = cmap, vmin = 0.001, vmax = maxValue)
    mat = ax.pcolormesh(lt, lat, data, vmin = 0.001, cmap = cmap)

    # ax.set_theta_direction(-1)  
    ax.set_theta_zero_location("W")
    # mat = ax.matshow(data,
    #                  cmap = cmap,
    #                  vmin = 0.001,
    #                  vmax = maxValue,
    #                  extent =[xRange[0],
    #                           xRange[1], 
    #                           yRange[0],
    #                           yRange[1]],
    #                  interpolation ='nearest',
    #                  aspect='auto',
    #                  origin ='lower')
    # if tickPos == 'top':
    #     if moons is not None:
    #         plotMoonOrbits(ax, moons=moons,
    #                        hemisphere='North')
    #     ax.tick_params(right=False, bottom=False, top=True,
    #                    labelright=False, labelbottom=False,
    #                    labeltop=True,rotation=0)
    # else:
    #     if moons is not None:
    #         plotMoonOrbits(ax, moons=moons,
    #                        hemisphere='South')
    #     ax.tick_params(right=False, bottom=True, top=False,
    #                    labelright=False, labelbottom=True,
    #                    labeltop=False,rotation=0)
    #     ax.set_xlabel(xlabel, size=14)
    #     leg = ax.legend(loc='lower right', frameon=False, fontsize=12)
    #     plt.setp(leg.get_texts(), color='#828282')

    leg = ax.legend(loc='lower right', frameon=True, fontsize=12)
    plt.setp(leg.get_texts(), color='#828282')

    ax.axvline(x = 180*np.pi/180, color='white', ls='--', alpha=0.6, lw=1)
    # ax.set_ylabel(ylabel, size=14, color='#b3b3b3')
    ax.grid(color='white', linestyle='--', linewidth=0.9, alpha=0.2)
    # ax.xaxis.set_major_locator(MultipleLocator(2))
    # ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_facecolor('#171717')
    return mat

def plotHeatmapPolar(matrix,
                xRange = [0, 24],
                yRange = [-90, 90],
                splitRange = None,
                splitBins = None,
                config = None,
                margin = None,
                suptitle = 'Cassini Cumulative Dwell Time along ' +
                           'Different Flux Tubes\n',
                cbar_label = 'Cumulative Dwell Time (Days)'):

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12,8))
    maxValue = np.max(matrix)

    ax1 = fig.add_subplot(121, polar='True')
    caxes1 = plotHeatmapPolarAxis(ax1, matrix, bins = splitBins[0],
                                               xRange = xRange,
                                               yRange = splitRange[0],
                                               tickPos = 'top')
    ax2 = fig.add_subplot(122, polar='True')
    caxes2 = plotHeatmapPolarAxis(ax2, matrix, bins = splitBins[1],
                                               xRange = xRange,
                                               yRange = splitRange[1],
                                               tickPos = 'bottom')
    caxes = caxes1

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.81, 0.1, 0.03, 0.78])
    cbar = fig.colorbar(caxes, cax=cbar_ax)
    cbar.set_label(label = cbar_label,
                   size = 14,
                   color = '#b3b3b3')
    fig.suptitle(suptitle, size=20, color='#787878')
    fig.subplots_adjust(wspace=0, hspace=0.06)
    fig.set_facecolor('#171717')

    # labels = [item.get_text() for item in ax2.get_xticklabels()]
    # print(labels)
    # labels[0] = '24'
    # ax2.set_xticklabels(labels)
    # axes.xaxis.label.set_size(20)
    # [t.set_color('red') for t in ax1.xaxis.get_ticklines()]
    # [t.set_color('red') for t in ax1.xaxis.get_ticklabels()]
    # ax2.get_xticklabels()[1].set_label("grey")
    # labels = [item.get_text() for item in ax2.get_xticklabels()]

    # ax1.get_xticklabels()[1].set_color("grey")
    # ax1.get_xticklabels()[-2].set_color("grey")
    # ax1.xaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))

    # labels = ax1.get_xticks().tolist()
    # labels[1] = '22'
    # labels[-2] = '2'
    # ax1.set_xticklabels(labels)

    # ax2.xaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))
    # ax2.get_xticklabels()[1].set_color("grey")
    # ax2.get_xticklabels()[-2].set_color("grey")

    # labels = ax2.get_xticks().tolist()
    # labels[1] = '22'
    # labels[-2] = '2'
    # ax2.set_xticklabels(labels)

    if config:
        if config.save:
            # path = os.path.join(ioconfig.path, 'heatmap'
            path = os.path.join('Output', 'heatmap' + '_%04d' % config.id +
                                config.format)
            plt.savefig(path, facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close()
        else:
            plt.show()
    else:
        plt.show()




def figure_output(fig, config={}):
    """ Save fig according to the config parameters in the dict, or 
    display the fig is no config is provided """
    if config.get("save"):

        if config.get("filename"):
            filename = config["filename"]
        else:
            filename = 'figure'

        if config.get("format"):
            fig_format = config["format"]
        else:
            fig_format = ".PNG"

        if config.get("id"):
            filename = filename+'_{:03d}{:s}'.format(config["id"], fig_format)
        else:
            filename = filename + '{:s}'.format(fig_format)

        if config.get("folder"):
            fullpath = os.path.join(config["folder"], filename)
        else:
            fullpath = filename

        if config.get("transparent"):
            fig.savefig(fullpath, transparent=True)
        else:
            #  fig.savefig(fullpath, facecolor=fig.get_facecolor())
            fig.savefig(fullpath)
        plt.close(fig)
    else:
        plt.show()


def styleGlobal(font_size=14, config={}):
    """ Global style parameters for all figures """
    #  plt.style.use('science')
    #  plt.style.use(['science','nature'])

    if config.get("theme"):
        theme_name = config["theme"]
        theme = config["themes"][theme_name]
        text_color = theme["font_color"]
        bg_color = theme["bg_color"]
        grid_color = theme["grid_color"]
    else:
        text_color = 'black'
        bg_color = 'white'
        grid_color = '#CCCCCC'

    # Edit the font, font size, and axes width
    f.rcParams['font.family'] = 'Avenir'
    f.rcParams['font.size'] = font_size
    f.rcParams['axes.linewidth'] = 1

    f.rcParams['text.color'] = text_color
    f.rcParams['axes.labelcolor'] = text_color
    f.rcParams['xtick.color'] = text_color
    f.rcParams['ytick.color'] = text_color

    f.rcParams['axes.facecolor'] = bg_color
    f.rcParams['figure.facecolor'] = bg_color
    f.rcParams['savefig.facecolor'] = bg_color
    f.rcParams['grid.color'] = grid_color
    f.rcParams['axes.grid'] = True


    #  mpl.rcParams['axes.edgecolor'] = grey
    #  mpl.rcParams['xtick.color'] = grey
    #  mpl.rcParams['ytick.color'] = grey
    #  mpl.rcParams['axes.labelcolor'] = "black"

    #  #  Try Seabormpl.rcParams['axes.linewidth'] = 0.3
    #  import seaborn as sns
    #  sns.set()

    # For processing in Adobe IL
    #  mpl.rcParams['pdf.fonttype'] = 42
    #  mpl.rcParams['ps.fonttype'] = 42
    #  mpl.rcParams['font.family'] = 'Arial'

    #  config.theme = "xkcd"
    #  if config.get("theme") == 'xkcd':
        #  scale, length, randomness = 1.1, 1.2, 0.1
        #  mpl.rcParams['font.family'] = ['Humor Sans', 'Comic Sans MS']
        #  mpl.rcParams['font.size'] = 14.0
        #  mpl.rcParams['path.sketch'] = (scale, length, randomness)
        #  mpl.rcParams['path.effects'] = [
            #  patheffects.withStroke(linewidth=4, foreground="w")]
        #  mpl.rcParams['axes.linewidth'] = 1.5
        #  mpl.rcParams['lines.linewidth'] = 2.0
        #  mpl.rcParams['figure.facecolor'] = 'white'
        #  mpl.rcParams['grid.linewidth'] = 0.0
        #  mpl.rcParams['axes.unicode_minus'] = False
        #  mpl.rcParams['axes.color_cycle'] = ['b', 'r', 'c', 'm']
        #  mpl.rcParams['xtick.major.size'] = 8
        #  mpl.rcParams['xtick.major.width'] = 3
        #  mpl.rcParams['ytick.major.size'] = 8
        #  mpl.rcParams['ytick.major.width'] = 3


def styleAxes(ax, config={}, fig=None):
    """ Style the ax axis by the config dict provided (or defaults if none
    provided) """

    #  mpl.rcParams['axes.spines.left'] = False
    #  mpl.rcParams['axes.spines.right'] = False
    #  mpl.rcParams['axes.spines.top'] = False
    #  mpl.rcParams['axes.spines.bottom'] = False

    theme_name = config["theme"]
    theme = config["themes"][theme_name]

    #  if theme_name == 'dark':
        #  plt.style.use('dark_background')

    #  if fig is not None:
        #  fig.set_facecolor(theme["bg_color"])
    #  ax.set_facecolor(theme["bg_color"])

    if config.get("xlim"):
        ax.set_xlim(config["xlim"])
    if config.get("ylim"):
        ax.set_ylim(config["ylim"])
    if config.get("xlabel"):
        ax.set_xlabel(config["xlabel"],
                labelpad=5,
                #  fontsize=12,
                color=theme["font_color"])
    if config.get("ylabel"):
        ax.set_ylabel(config["ylabel"],
                labelpad=5,
                #  fontsize=12,
                color=theme["font_color"])
    if config.get("title"):
        ax.set_title(config["title"], color=theme["font_color"])

    # Subjective stylistic choises
    ax.grid(color=theme["grid_color"],
            linestyle=theme["grid_style"],
            linewidth=theme["grid_width"],
            alpha=theme["grid_alpha"])
    #  b=True, which='minor'
    ax.tick_params(axis='x', colors=theme["font_color"], direction='in')
    ax.tick_params(axis='y', colors=theme["font_color"], direction='in')
    ax.tick_params(direction="in")
    # ax.yaxis.set_tick_params | which='major', size=8, width=1
    ax.spines['bottom'].set_color('0.5')
    ax.spines['top'].set_color('0.5')
    ax.spines['right'].set_color('0.5')
    ax.spines['left'].set_color('0.5')

    if config.get("major_locator"):
        ax.xaxis.set_major_locator(MultipleLocator(config["major_locator"]))
    if config.get("minor_locator"):
        ax.xaxis.set_major_locator(MultipleLocator(config["minor_locator"]))
    if config.get("equal_aspect"):
        ax.set_aspect('equal', 'box')
    if config.get("legend"):
        #  ax.legend(frameon=False)
        ax.legend(fancybox=True, framealpha=0.1)

def styleColorbar(ax, obj, ticks=None, barlabel=None, ticklabels=None, location='right'):

        cbar = plt.colorbar(obj, ax=ax,
                location=location, ticks=ticks, pad=0.02, extend='both',
                                extendfrac=0.05, fraction=0.046)
        if barlabel is not None:
            cbar.set_label(barlabel, labelpad=-5)
        if ticklabels is not None:
            cbar.ax.set_yticklabels(ticklabels)

        # Create a new axis for the colorbar
        #  divider = make_axes_locatable(ax)
        #  cax = divider.append_axes("right", size="3%", pad=0.03)
        #  cbar = fig.colorbar(s, cax=ax, ticks=[1,10]) #| fraction=0.046, pad=0.04

        #  cbar.ax.tick_params(labelsize=10, direction='in', which='both', width=1.5, length=7, color='white', grid_alpha=0.2)
        #  cbar.ax.tick_params(labelsize=10, direction='in', which='minor', width=0.7, length=4, color='white', grid_alpha=0.2)
        #  cbar.ax.ticklabel_format(axis='both')
        return cbar


def plotSurfaceMappingMeridian(fls, flattening=0, config=None, time=None):

    styleGlobal(font_size=12, config=config)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=config["dpi"])

    theta = np.linspace( 0 , 2 * np.pi , 150 )
    radius = 1.
    a = radius * np.cos( theta )
    b = radius * np.sin( theta )
    # ax.plot( a, b, color='white', ls='--', alpha=0.7)
    ax.fill( a, b, 'white', alpha=0.2, label='Spherical Saturn')
    ax.set_aspect( 1 )

    a = radius * np.cos( theta )
    b = radius * (1 - flattening) * np.sin( theta )
    ax.plot( a, b, color='yellow', alpha=0.8, label='Oblate Spheroidal Saturn')
    ax.set_aspect( 1 )

    cmap = copy.copy(plt.cm.get_cmap("inferno")) # jet, OrRd, magma, plasma, inferno
    cmap.set_under(color='black')

    for fl in fls:
        X = np.asarray(fl.traceXYZ[:,0])
        Y = np.asarray(fl.traceXYZ[:,1])
        Z = np.asarray(fl.traceXYZ[:,2])
        # RHO = np.sqrt(np.power(X, 2)+ np.power(Y, 2))
        ax.plot(X, Z,
                ls='--', lw=1, c='white')

        s = ax.scatter(X, Z,
                        s=15, c='white')
                        #  s=15, c=fl.dwelltime, cmap=cmap, vmin=0.0001)

        # s = ax.scatter(X[:], Z[:],
                        # s=15, c=fl.dwelltime[:], cmap=cmap, vmin=0.0001)
                        # label = label, s=10, c=fieldline.dwelltime[:], cmap=my_cmap,
                        # norm=LogNorm(vmin=vmin, vmax=vmax), alpha=1.0, zorder=1E6)

        ax.scatter(X[0], Z[0],
                s=10, alpha=0.8, marker="o", c="yellow") #c='#00cc8f'
        ax.scatter(X[-1], Z[-1],
                s=10, alpha=0.8, marker="o", c="#00cc8f")


    cbar = styleColorbar(ax, s, ticks=None, barlabel=config["title"],
                         ticklabels=None, location='right')
    #  fig.subplots_adjust(right=0.8)
    #  cbar_ax = fig.add_axes([0.81, 0.1, 0.03, 0.78])
    #  cbar = fig.colorbar(s, cax=cbar_ax)
    #  cbar.set_label(label = config["title"],
                   #  size = 14,
                   #  color = '#b3b3b3')

    styleAxes(ax, config, fig=fig)
    fig.tight_layout()
    figure_output(fig, config)

def plotRDistribution(RMap):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)

    R_axis = np.linspace(0, 50, size(RMap))
    plt.plot(R_axis, RMap/60/60/24)

    ax.set_xlabel(r'r ($R_S$)', size=14)
    ax.set_ylabel(r'Cumulative Time (days)', size=14)
    # ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.2)
    fig.set_facecolor('#171717')
    ax.set_facecolor('#171717')
    plt.show()

# Simulate Symmetrical Orbit Data for Testing Tracing to the Surface 
def simulateOrbitData(dateFrom='2005-02-01T00:00:00', ntimes = 1, nphi=5, nr=2, rstart=None):
    date0 = datetime.datetime.strptime(dateFrom, DATETIME_FORMAT[0])
    Time = []
    coords = []
    if rstart is not None:
        r = rstart
        # coords += [[r*np.cos(phi),r*np.sin(phi),0.] for phi in np.linspace(0+np.pi/10, 2*np.pi, nphi)]
        coords += [[r*np.cos(phi),r*np.sin(phi),0.] for phi in np.linspace(0, 2*np.pi*ntimes, nphi*ntimes)]
#        Time = [date0 + datetime.timedelta(days=15.945/nphi)*n for n in range(0, nphi*ntimes)]
        Time = [date0 + datetime.timedelta(seconds=60*60*60/nphi)*n for n in range(0, nphi*ntimes)]
    else:
        for i in range(0, nr+1):
            r = 1.2 + 5 * i
            coords += [[r*np.cos(phi),r*np.sin(phi),0.] for phi in np.linspace(0+np.pi/10, 2*np.pi, nphi)]
        Time = [date0 + datetime.timedelta(seconds=60)*n for n in range(0, nphi*(nr-1))]
    coords = np.asarray(coords).transpose()
    # plotSurfaceMapping(fls, flattening=FLATTENING, time=Time[0])
    return Time, coords


def plotLTSectors_AlongField(data_list, xaxis, config=None, title_list=None, suptitle=None):
    """ Plot dwell times or event to dwell time ratio along the field
    with the provided binAxis.
    Arguments:
        data_list : list(M) of array(N) (event time or ratio of event to dwell time)
        xaxis: array(N) (center locations of bins in lat)
        config: list(M) of dict (figure and axis style parameters)
    """

    #  with plt.xkcd():
    styleGlobal(font_size=12, config=config)
    fig, axes = plt.subplots(2,2, figsize=(8, 5), dpi=config.dpi)
    #  fig.subplots_adjust(left=0.2, wspace=0.6)

    color = config["color"] if config.get("color") else "#f79216"
    label = config.get("label")
    alpha = config.get("alpha")

    for ax, data, title in zip(axes.flat, data_list, title_list):
        label = title
        config.title = title
        config.legend = False
        ax.plot(xaxis, data, color=color, alpha=alpha, label=label)
        styleAxes(ax, config, fig=fig)
        #  ax.legend(frameon=False)

    if suptitle is not None:
        fig.suptitle(suptitle)

    fig.tight_layout()
    fig.align_ylabels() # Optional: pass axes to align, e.g. axes[:,1]

    figure_output(fig, config)


def plotLatitudeBinAll(data_list, xaxis, config_list=[], suptitle=None):
    """ Plot dwell times or event to dwell time ratio along the field
    with the provided binAxis.
    Arguments:
        data_list : list(M) of array(N) (event time or ratio of event to dwell time)
        xaxis: array(N) (center locations of bins in lat)
        config: list(M) of dict (figure and axis style parameters)
    """

    styleGlobal(font_size=12, config=config_list[0])
    fig, axes = plt.subplots(3,1, figsize=(7, 6), dpi=config_list[0].dpi)
    #  fig.subplots_adjust(left=0.2, wspace=0.6)

    for ax, data, config in zip(axes, data_list, config_list):
        color = config["color"] if config.get("color") else "#f79216"
        label = config.get("label")
        alpha = config.get("alpha")
        ax.plot(xaxis, data, color=color, alpha=alpha, label=label)
        styleAxes(ax, config, fig=fig)
        ax.legend(frameon=False)

    if suptitle is not None:
        fig.suptitle(suptitle)

    fig.tight_layout()
    fig.align_ylabels() # Optional: pass axes to align, e.g. axes[:,1]

    figure_output(fig, config)

def plotLatitudeBinTimes(data, xaxis, config={}, fig=None, ax=None):
    """ Plot dwell times or event to dwell time ratio along the field
    with the provided binAxis.
    Arguments:
        data : array(N) (event time or ratio of event to dwell time)
        xaxis: array(N) (center locations of bins in lat)
        config: dict (figure and axis style parameters)
        ax: axis object (optional)
        fig: parent figure object (optional)
    """

    newFigure = True if ax is None else False


    if newFigure:
        styleGlobal(config=config)
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    color = config["color"] if config.get("color") else "#f79216"
    label = config.get("label")
    alpha = config.get("alpha")
    ax.plot(xaxis, data, color=color, alpha=alpha, label=label)

    styleAxes(ax, config, fig=fig)

    if newFigure:
        fig.tight_layout()
        figure_output(fig, config)


