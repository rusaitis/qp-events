import os
from sys import path
from os.path import isfile, join
import numpy as np
import datetime
from datetime import timezone
import cassinilib
from cassinilib.DataPaths import *
from cassinilib.NewSignal import *
from cassinilib.Plot import *
from cassinilib.DatetimeFunctions import *
from wavesolver.sim import *
from wavesolver.fieldline import *
from wavesolver.io import *
from wavesolver.linalg import *
from KMAGhelper.KmagFunctions import *
from wavesolver.configurations import *
from wavesolver.helperFunctions import *
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import copy


class interval():
    """ Class to store interval objects with comments """ 
    def __init__(self, dateFrom=None, dateTo=None, comment=None):
        """ Interval class to keep track of data discontinuities """
        self.dateFrom = dateFrom
        self.dateTo = dateTo
        self.comment = comment


def filterDataSpatially(data, Lat=None, LT=None, Mlat=None):
    """ Filter 3D np.array of cumulative times by Inv Lat, LT, and Mag Lat
    Arguments:
        data: (np.array, 3D)
        Lat: (list of size 2) invariant lat of the selection, -90 - 90 [deg]
        LT: (list of size 2) local time of the selection, 0-24 [h]
        Mlat: (list of size 2) magnetic lat of the selection, -90 - 90[deg]
    """
    if Lat is not None:
        Nbins = len(data[:,0,0])
        bin_i = value2bin(Lat[0], -90, 90, bins=Nbins, range=True)
        bin_j = value2bin(Lat[1], -90, 90, bins=Nbins, range=True)
        data = data[bin_i:bin_j, :, :]

    if LT is not None:
        # handle LT selection if it falls beyond [0,24] range
        LT_list = []
        if LT[0] < 0:
            LT_list.append([24-abs(LT[0]), 24   ])
            LT_list.append([0            , LT[1]])
        elif LT[1] > 24:
            LT_list.append([LT[0], 24      ])
            LT_list.append([0    , LT[1]-24])
        else:
            LT_list.append(LT)

        data_temp = np.zeros_like(data)
        for LT in LT_list:
            Nbins = len(data[0,:,0])
            bin_i = value2bin(LT[0], 0, 24, bins=Nbins, range=True)
            bin_j = value2bin(LT[1], 0, 24, bins=Nbins, range=True)
            data_temp[:, bin_i:bin_j, :] += data[:, bin_i:bin_j, :]
        data = data_temp

    if Mlat is not None:
        Nbins = len(data[0,0,:])
        bin_i = value2bin(Lat[0], -90, 90, bins=Nbins, range=True)
        bin_j = value2bin(Lat[1], -90, 90, bins=Nbins, range=True)
        data = data[:, :, bin_i:bin_j]
    return data

def reduceToMlat(data):
    """ Reduce a 3-d np.array of cumulative times to the Mag Lat axis """
    #  return np.sum(np.sum(data, axis=1), axis=0)
    return np.sum(np.sum(data, axis=0), axis=0)
    #  aa = np.sum(data, axis=0)
    #  return np.sum(aa, axis=0)

def constructFieldLines(dataIn,
                        LatBinsSelection, # Number of Lat Bins (int)
                        LatSelectionFrom,
                        LatSelectionTo,
                        LTSelectionFrom,
                        LTSelectionTo,
                        r=1.1,
                        phi=0.,
                        maxIter=None,
                        LatBins=18,
                        LTBins=24,
                        MagLatBins=36,
                        SIM=None):
  fls = []
  for i in range(0, LatBinsSelection+1):
    data = copy.deepcopy(dataIn)
    lat = bin2value(i,
                    minValue = np.radians(LatSelectionFrom),
                    maxValue = np.radians(LatSelectionTo),
                    bins = LatBinsSelection)

    #  lat = np.radians(80)
    fl = traceFieldline(sph2car([r, Colat2Lat(lat), phi]), SIM)
    #  fl = traceFieldline(sph2car([r, lat, phi]), SIM)

    # Select Specified Data Range
    LTBinFrom = value2bin(LTSelectionFrom, 0, 24, bins = LTBins)
    LTBinTo = value2bin(LTSelectionTo, 0, 24, bins = LTBins)
    LatBinFrom = value2bin(lat, -np.pi/2, np.pi/2, bins = LatBins) - 1
    LatBinTo = LatBinFrom + 1
    data = data[:, LTBinFrom:LTBinTo, :]
    data = data[LatBinFrom:LatBinTo, :, :]

    # Integrate over other axes
    data = np.sum(data, axis=1)
    data = np.sum(data, axis=0)

    # Normalize to hours
    data = data / 60. / 60./ 24.

    # Save the dwell time in a LT-InvLat bin as array of dwell times vs MagLat
    # along a typical field line in that bin
    fl.dwelltime = data[value2bin(Colat2Lat(fl.traceRTP[:,1]),
                                  minValue = -np.pi/2,
                                  maxValue = np.pi/2,
                                  bins = MagLatBins)]
    fls.append(fl)
    if maxIter is not None:
        if i > maxIter:
          break
    # Return all typical field lines corresponding to the data map selection
  return fls


def readEventTimes(conf):
    """ Load event time data from a saved Numpy array """
    path = os.path.join(conf.data_dir, conf.event_time.folder)
    found_files = [f for f in os.listdir(path) if
                   (f.endswith('.npy') and f.startswith(conf.data_name))]
    data = np.load(os.path.join(path, found_files[0]))
    return data

def readDwellTimes(conf):
    """ Load dwell time data from a saved Numpy array """

    filename_start = conf.data_name

    # Initialize Binning Arrays
    SurfaceMap = np.zeros((conf.nbins.Lat,
                           conf.nbins.LT,
                           conf.nbins.Mlat))

    RMap = np.zeros(conf.nbins.R)

    if conf.dwell_time.split_by_year:
        n = 0
        saveid = 0
        for year in conf.data_years:
            path = os.path.join(conf.data_dir,
                                conf.dwell_time.folder,
                                str(year))

            # allfiles = [f for f in os.listdir(ioconfig.path) if
            #  os.path.isfile(os.path.join(ioconfig.path, f))]
            found_files = [f for f in os.listdir(path) if
                           (f.endswith('.npy')
                            and f.startswith(filename_start))]

            # n = 0
            # found_files = [f for f in os.listdir(ioconfig.path) if (f.endswith('.npy') and f.startswith(filename_start))]
            # lookingForFiles = True
            # while lookingForFiles:
            #   filename_start0 = 'surfaceMap_%05d' % n
            #   # print(filename_start0)
            #   found_file2 = [f for f in os.listdir(ioconfig.path) if (f.endswith('.npy') and f.startswith(filename_start0))]
            #   # print(found_file2)
            #   if found_file2:
            #     filename = os.path.splitext(found_file2[0])[0].split('_')
            #     ioconfig.name = ''
            #     ioconfig.id = saveid
            #     saveid += 1
            #     if 'to' in filename:
            #       dateFrom = dateFrom0
            #       dateTo = filename[-1]
            #       ioconfig.time = dateFrom + ' - ' + dateTo
            #     else:
            #       ioconfig.time = filename[-1]

            # SurfaceMap_temp = np.load(os.path.join(ioconfig.path, found_file2[0]))
            # SurfaceMap += SurfaceMap_temp
            #     plotHeatmap(np.sum(SurfaceMap, axis=2),
            #         bins=(LatBins, LTBins),
            #         config=ioconfig)
            #     # if n > 10: break
            #   else:
            #     lookingForFiles = False
            #   n += 1

            SurfaceMap_ = np.load(os.path.join(path, found_files[0]))
            SurfaceMap += SurfaceMap_
            # processMovie2('Output/heatmap_%04d.png', res='5136x2880', fps=30, output='movie.mp4', config=ioconfig)
    return SurfaceMap

if __name__ == "__main__":
# Command line aguments
    parser = argparse.ArgumentParser(description='Process Event/Dwell Times')
    parser.add_argument('--calc', dest='calculate', action='store_true',
                      help='Compute the Alfven waves')
    parser.add_argument('--mov', dest='movie',
                      action='store_true', help='Generate a movie')
    parser.add_argument('--save', dest='save',
                      action='store_true', help='Save the figures')
    parser.add_argument('--plot', dest='plot',
                      action='store_true', help='Plot the final surface map')
    parser.add_argument('--saveProgress', dest='saveProgress',
                      action='store_true', help='Save progress')
    parser.add_argument('--dipole', dest='dipole',
                      action='store_true', help='Use a dipole field')
    parser.add_argument('--fast', dest='fast', action='store_true',
                      help='Enable a faster calculation (less accurate)')
    parser.add_argument('--step', dest='step', type=float,
                      default=0.1, nargs=1,
                      help='Field line tracing step size (in planet radii)')
    parser.add_argument('--output', dest='output', type=str,
                      help='Custom output folder')
    parser.add_argument('--iter', dest='iter', type=int, default=0,
                      help='Iteration Number')
    parser.add_argument('--Lat', dest='Lat', type=int, nargs = 2,
                      help='Invariant Latitude of the Field Line Data')
    parser.add_argument('--LT', dest='LT', type=int, nargs = 2,
                      help='Invariant Latitude of the Field Line Data')
    parser.add_argument('--Mlat', dest='Mlat', type=int, nargs = 2,
                      help='Invariant Latitude of the Field Line Data')
    parser.add_argument('--R', dest='R', type=int, nargs = 2,
                      help='Invariant Latitude of the Field Line Data')
    parser.add_argument('--plottype', dest='plottype', type=str, default='ratio',
                      help='Type of plot to show')
    parser.add_argument('--min_dwell', dest='min_dwell', type=int, default=1,
                      help='Minimum Spacecraft Dwell Time in each Bin')
    args = parser.parse_args()

    # Set up the output folder and other plotting parameters
    if args.output is not None:
        customFolder = args.output
        outputPath = os.path.join('Output', customFolder)
        createDirectory(outputPath)
    else:
        outputPath = 'Output'

    ioconfig = io(path=outputPath,
                  save=args.save,
                  movie=args.movie,
                  format='.png')

    figs = {"heatmap": {"cmap_range": [0.0001, 25],
                        "split_lat": 60,
                        "xlabel": r'Magnetic Latitude [deg]',
                        "ylabel": r'Conjugate Latitude [deg]',
                        "traces": None,
                        "title": None,
                       },
            "dwell_time_along_field": {
                "title": None,
                "xlabel": r'Magnetic Latitude [deg]',
                "ylabel": r'Dwell Time [days]',
                "label": r'Dwell Time [days]',
                "xlim": [-90, 90],
                "ylim": None,
                "legend": True,
                "major_locator": 20,
            },
            "event_time_along_field": {
                "title": None,
                "xlabel": r'Magnetic Latitude [deg]',
                "ylabel": r'Event Time [days]',
                "label": r'Event Time [days]',
                "xlim": [-90, 90],
                "ylim": None,
                "legend": True,
                "major_locator": 20,
            },
            "ratio_along_field": {
                "title": None,
                "xlabel": r'Magnetic Latitude [deg]',
                "ylabel": r'Ratio',
                "label": r'Ratio of Event to Dwell Time',
                "xlim": [-90, 90],
                "ylim": [0, 0.3],
                "legend": True,
                "major_locator": 20,
            },
            "meridian_field_lines": {
                "title": None,
                "xlabel": r'x [$R_S$]',
                "ylabel": r'z [$R_S$]',
                "label": r'Ratio of Event to Dwell Time',
                "xlim": [-20, 20],
                "ylim": [-20, 20],
                "legend": False,
                "major_locator": 5,
            },
            }

    ioconfig_dict={"data_dir": "/Users/leo/DATA",
                   "event_time": {"folder": 'cassini-sc-event-time-north',
                                  "split_by_year": False,
                                  },
                   "dwell_time": {"folder": 'cassini-sc-dwell-time-plasma',
                                  #  "split_by_year": True,
                                  "split_by_year": False,
                                  "min_dwell_days": 0.1,
                                  },
                   "selection": {"Lat": [0, 90],
                                 "LT": [8, 16],
                                 "Mlat": [-90, 90],
                                 "R": [1, 50],
                                 },
                   "calculation": {},
                   "orbit": {"folder": 'cassini-orbit',
                             },
                   "coord": "KSM",
                   "data_name": "surfaceMapOpen",
                   "data_years": [y for y in range(2004, 2017+1)],
                   "id": None,
                   "nbins": {"Lat": 2 * 90,
                             "LT": 2 * 24,
                             "Mlat": 18 * 2,
                             "R": 50},
                   "plot": {"format": 'PNG',
                            "dpi": 150,
                            "figsize": [7,3],
                            "title": None,
                            "suptitle": None,
                            "movie": False,
                            "save": False,
                            "folder": 'Output',
                            "filename": 'figure',
                            "transparent": False,
                            "theme": 'dark',
                            "themes": {
                                "dark": {
                                    "bg_color": "#191919",
                                    "font_color": "#ffffff",
                                    "grid_color": "#ffffff",
                                    "grid_alpha": 0.3,
                                    "grid_style": ":",
                                    "grid_width": 1.,
                                },
                                "light": {
                                    "bg_color": "#ffffff",
                                    "font_color": "#171717",
                                    "grid_color": "#CCCCCC",
                                    "grid_alpha": 0.5,
                                    "grid_style": ":",
                                    "grid_width": 1.,
                                },
                            },
                            "equal_aspect": None,
                            },
                   "figs": figs,
                   "run": {"heatmap": True,
                           "event_time_along_field": False,
                           "dwell_time_along_field": False,
                           "ratio_along_field": False,
                           "all_along_field": False,
                           "LT_sectors_ratio_along_field": False,
                           }
                  }


    ioconfig = Config(ioconfig_dict)

    # Data Selection by CLI arguments
    if args.Lat is not None:
        ioconfig.selection.Lat = args.Lat
    if args.LT is not None:
        ioconfig.selection.LT = args.LT
    if args.Mlat is not None:
        ioconfig.selection.Mlat = args.Mlat
    if args.R is not None:
        ioconfig.selection.R = args.R
    
    # Adjust DPI for Publishing Quality if Saving
    ioconfig.plot.dpi = 300 if ioconfig.plot.save else 150

    title_dict = {"surfaceMap" : 'All Data',
                  "surfaceMapClosed": "Closed Field Lines",
                  "surfaceManOpen": "Open Field Lines",
                  "surfaceMapNearby": r'Nearby Events Only',
                  }

    ioconfig.plot.title = switch(title_dict, ioconfig.data_name)

    SurfaceMap = readDwellTimes(ioconfig)
    SurfaceMapEvents = readEventTimes(ioconfig)

    # Normalize the data to days
    SurfaceMap = SurfaceMap/(24.*60.*60.)
    SurfaceMapEvents = SurfaceMapEvents/(24.*60.*60.)

    # Reduce to 2D by integrating the last axis (Mag Lat)
    SurfaceMap_2D = np.sum(SurfaceMap, axis=2)
    SurfaceMapEvents_2D = np.sum(SurfaceMapEvents, axis=2)

    # Calculate the Event to Dwell Time Ratio
    EventRatio = np.divide(SurfaceMapEvents_2D, SurfaceMap_2D,
                           dtype=float,
                           out=np.zeros_like(SurfaceMap_2D),
                           where=SurfaceMap_2D > ioconfig.dwell_time.min_dwell_days,
                           )
    print(np.max(SurfaceMap_2D))
    print(np.max(SurfaceMapEvents_2D))
    exit()

    plotHeatmap2D = True
    if plotHeatmap2D:
        heatmapRange = [0.0001, 0.5]
        #  heatmapRange = [0.0001, 25]
        heatmap_split_Lat = 0
        sl = 0
        #  heatmap = EventRatio
        #  heatmap = SurfaceMap_2D
        heatmap = SurfaceMapEvents_2D

        plotHeatmap(heatmap,
                  colorRange = heatmapRange,
                  xRange = [0, 24],
                  yRange = [-90, 90],
                  splitRange = [[ sl,  90],
                                [-90, -sl]],
                  splitBins=[[value2bin( sl, minValue=-90, maxValue=90, bins=ioconfig.nbins.Lat, range=True),
                              value2bin( 90, minValue=-90, maxValue=90, bins=ioconfig.nbins.Lat, range=True)],
                             [value2bin(-90, minValue=-90, maxValue=90, bins=ioconfig.nbins.Lat, range=True),
                              value2bin(-sl, minValue=-90, maxValue=90, bins=ioconfig.nbins.Lat, range=True)]],
                  margin=value2bin( 2, minValue=0, maxValue=24, bins=ioconfig.nbins.LT),
                  config=ioconfig)
        exit()

    def selectDataAlongField(DwellMap, EventMap, ioconfig):
        # Select the desired portion of the data (by LT and Latitude)
        DwellMap = filterDataSpatially(DwellMap,
                                       Lat=ioconfig.selection.Lat,
                                       LT=ioconfig.selection.LT)
        EventMap = filterDataSpatially(EventMap,
                                       Lat=ioconfig.selection.Lat,
                                       LT=ioconfig.selection.LT)

        # Sum over the other axes of the data to get times for the Mag Latitudes
        t_dwell = reduceToMlat(DwellMap)
        t_event = reduceToMlat(EventMap)

        # Calculate the Event to Dwell Time Ratio
        ratio = np.divide(t_event,
                          t_dwell,
                          dtype=float,
                          out=np.zeros_like(t_dwell),
                          where=t_dwell > ioconfig.dwell_time.min_dwell_days)
        return t_dwell, t_event, ratio

    time_along_field, time_event_along_field, event_ratio = \
        selectDataAlongField(SurfaceMap, SurfaceMapEvents, ioconfig)

    data = [time_along_field,
            time_event_along_field,
            event_ratio]

    fig_configs = [Config(ioconfig.plot | ioconfig.figs.dwell_time_along_field),
                   Config(ioconfig.plot | ioconfig.figs.event_time_along_field),
                   Config(ioconfig.plot | ioconfig.figs.ratio_along_field)]


    xaxis = [bin2value(i, minValue=-90, maxValue=90, bins=ioconfig.nbins.Mlat)
             for i in arange(0, ioconfig.nbins.Mlat)]

    name_short = ioconfig.plot.title
    title = (r'$LT$: {:d}-{:d} h  |  '
             r'$\theta_{{inv}}$ = {:d}-{:d}$^{{\circ}}$  |  '
             '{:s}'.format(*ioconfig.selection.LT,
                           *ioconfig.selection.Lat,
                           name_short))

    if ioconfig.run.dwell_time_along_field:
        plotLatitudeBinTimes(data[0], xaxis, config=Config(ioconfig.plot | fig_configs[0]))
    if ioconfig.run.event_time_along_field:
        plotLatitudeBinTimes(data[1], xaxis, config=Config(ioconfig.plot | fig_configs[1]))
    if ioconfig.run.ratio_along_field:
        plotLatitudeBinTimes(data[2], xaxis, config=Config(ioconfig.plot | fig_configs[2]))
    if ioconfig.run.all_along_field:
        plotLatitudeBinAll(data, xaxis, config_list=fig_configs, suptitle=title)
    if ioconfig.run.LT_sectors_ratio_along_field:
        LT_list = [[-3,3], [3,9], [9,15], [15, 21]]
        title_list = ['Midnight', 'Dawn', 'Noon', 'Dusk']
        title_list = [t + r' $\pm$ 3h' for t in title_list]
        data = []
        for LT in LT_list:
            ioconfig.selection.LT = LT
            time_along_field, time_event_along_field, event_ratio = \
                selectDataAlongField(SurfaceMap, SurfaceMapEvents, ioconfig)
            data.append(event_ratio)
        plotLTSectors_AlongField(data, xaxis, config=fig_configs[2],
                                 title_list=title_list,
                                 suptitle='Ratio of Event to Dwell Time')

    SIM = loadsim(configSaturnNominal)
    # SIM = loadsim(Saturn)
    # SIM.config["ETIME"] = 1483228800  # (01 Jan 2017)
    SIM.config["ETIME"] = 1483228900  # (01 Jan 2017)
    SIM.config["ETIME"] = 1250035200  # (12 Aug 2009 - Equinox )
    # SIM.config["BFieldModel"] = dipField  # (01 Jan 2017)
    # SIM.config["BFieldModel"] = KMAGField  # (01 Jan 2017)
    SIM.config["IN_COORD"] = 'DIS'  # S3C/DIS/DIP/KSM/KSO
    SIM.config["OUT_COORD"] = 'DIS'  # S3C/DIS/DIP/KSM/KSO
    SIM.config["IN_CARSPH"] = 'CAR'  # INPUT IN SPH/CAR
    SIM.config["OUT_CARSPH"] = 'CAR'  # OUTPUT IN SPH/CAR
    SIM.config["step"] = 0.05
    SIM.config["maxIter"] = 1E6
    SIM.config["method"] = 'euler'
    SIM.config["maxR"] = 50


  ##  print(SurfaceMap.shape)
  ##  print(data.shape)
  ##  theta = np.linspace(0, 2*np.pi, size(data[:,0]))
  ##  print(theta.shape)
  ##  phi = np.linspace(0, 2*np.pi, size(data[0,:]))
  ##  print(phi.shape)
  ##  theta, phi = np.meshgrid(theta, phi)
  ##  # The Cartesian coordinates of the unit sphere
  ##  x = np.cos(theta) * np.cos(phi)
  ##  y = np.cos(theta) * np.sin(phi)
  ##  z = np.sin(theta)
  ##

    u, v = np.mgrid[0:2*np.pi:48j, 0:np.pi:180j]
    x = 1.1*np.cos(u)*np.sin(v)
    y = 1.1*np.sin(u)*np.sin(v)
    z = np.cos(v)

#
    #  a = radius * np.cos( theta )
    #  b = radius * (1 - flattening) * np.sin( theta )

    plt.style.use('dark_background')
    #  fig = plt.figure(figsize=plt.figaspect(1)*2)
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_subplot(projection='3d', proj_type='ortho') #ortho | persp
    ax.set_box_aspect([1,1,1])
    data = np.sum(SurfaceMap, axis=2)
    data = np.sum(SurfaceMapEvents, axis=2)

    #  drawSaturn3D(ax, draw_Plane=True, draw_Rings=True)
    c = 2
    pts = [[-c,-c,-c], [-c,c,-c], [c,c,-c], [c,-c,-c],
            [-c,-c,c], [-c,c,c], [c,c,c], [c,-c,c]]
    pts=np.asarray(pts).T
    #  ax.scatter(pts[0], pts[1], pts[2], color='grey')

    norm = matplotlib.colors.Normalize(vmin = 0.01, vmax=np.max(data))
    cmap = copy.copy(plt.cm.get_cmap("turbo")) # jet | OrRd plasma
    cmap.set_under(color='#090909')
    #  ax.plot_surface(x, y, z, rcount=48, ccount=180, facecolors=cmap(np.flip(norm(data.T), axis=1)), shade=False)
    ax.plot_surface(x, y, z, rcount=48, ccount=180, facecolors=cmap(norm(data.T)), shade=False, zorder=10)
    RefColatitude = 80
    drawOrbit(ax, R=(0,0), r=1*np.cos(RefColatitude * np.pi / 180.), color="#ffd000", alpha=0.3, rotate=False, linewidth=1, z=1*np.sin(RefColatitude * np.pi / 180.))
    drawOrbit(ax, R=(0,0), r=1*np.cos(RefColatitude * np.pi / 180.), color="#ffd000", alpha=0.3, rotate=False, linewidth=1, z=-1*np.sin(RefColatitude * np.pi / 180.))
    RefColatitude = 70
    drawOrbit(ax, R=(0,0), r=1*np.cos(RefColatitude * np.pi / 180.), color="#ffd000", alpha=0.3, rotate=False, linewidth=1, z=1*np.sin(RefColatitude * np.pi / 180.))
    drawOrbit(ax, R=(0,0), r=1*np.cos(RefColatitude * np.pi / 180.), color="#ffd000", alpha=0.3, rotate=False, linewidth=1, z=-1*np.sin(RefColatitude * np.pi / 180.))

  # Turn off the axis planes
    ax.set_facecolor('#171717')
    ax.set_xlabel(r'$x_{KSM}$ [$R_S$]')
    ax.set_ylabel(r'$y_{KSM}$ [$R_S$]')
    ax.set_zlabel(r'$z_{KSM}$ [$R_S$]')

    fig.set_facecolor('#171717')
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.05))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.05))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.05))
    ax.xaxis._axinfo["grid"].update({"linewidth":0.5,"linestyle":"--",'color':'#383838', 'alpha':0.4})
    ax.yaxis._axinfo["grid"].update({"linewidth":0.5,"linestyle":"--",'color':'#383838', 'alpha':0.4})
    ax.zaxis._axinfo["grid"].update({"linewidth":0.5,"linestyle":"--",'color':'#383838', 'alpha':0.4})
    ax.view_init(elev=30, azim=290)

    #  ax.set_box_aspect((1, 1, 1))
    #  PlotAxes(ax, axlim=None, color='white', alpha=0.6, lw=2, mutation_scale=12)
    #  ax.set_axis_off()

    # -------------------------------------
    LatBinsSelection = 4
    LatSelectionFrom = 60
    LatSelectionTo = 75
    LTSelectionFrom = 12 - 1
    LTSelectionTo = 12 + 1
    # SIM.BFieldModel = dipField
    SIM.BFieldModel = KMAGField
    fls = constructFieldLines(SurfaceMap, LatBinsSelection, LatSelectionFrom,
                              LatSelectionTo, LTSelectionFrom, LTSelectionTo,
                              r=1.02, phi=0, maxIter=10,
                              LatBins   = ioconfig.nbins.Lat,
                              LTBins    = ioconfig.nbins.LT,
                              MagLatBins= ioconfig.nbins.Mlat,
                              SIM=SIM)

    LTSelectionFrom = 0
    LTSelectionTo = 0 + 1
    fls2 = constructFieldLines(SurfaceMap, LatBinsSelection, LatSelectionFrom,
                               LatSelectionTo, LTSelectionFrom, LTSelectionTo,
                               r=1.02, phi=np.pi, maxIter=10,
                               LatBins   = ioconfig.nbins.Lat,
                               LTBins    = ioconfig.nbins.LT,
                               MagLatBins= ioconfig.nbins.Mlat,
                               SIM=SIM)
    fls.extend(fls2)

    LTSelectionFrom = 6-1
    LTSelectionTo = 6+1
    fls3 = constructFieldLines(SurfaceMap, LatBinsSelection, LatSelectionFrom,
                               LatSelectionTo, LTSelectionFrom, LTSelectionTo,
                               r=1.02, phi=np.pi/2, maxIter=10,
                               LatBins   = ioconfig.nbins.Lat,
                               LTBins    = ioconfig.nbins.LT,
                               MagLatBins= ioconfig.nbins.Mlat,
                               SIM=SIM)
    fls.extend(fls3)

    LTSelectionFrom = 18-1
    LTSelectionTo = 18+1
    fls4 = constructFieldLines(SurfaceMap, LatBinsSelection, LatSelectionFrom,
                               LatSelectionTo, LTSelectionFrom, LTSelectionTo,
                               r=1.02, phi=2*np.pi*3/4, maxIter=10,
                               LatBins   = ioconfig.nbins.Lat,
                               LTBins    = ioconfig.nbins.LT,
                               MagLatBins= ioconfig.nbins.Mlat,
                               SIM=SIM)
    fls.extend(fls4)
    for fl in fls:
        #  plt.scatter(X,Y,Z, color='white')
        ax.plot(fl.traceXYZ[0,0],
                fl.traceXYZ[0,1],
                fl.traceXYZ[0,2],
                ms=2, marker='.',
                alpha=0.8, color='yellow', zorder=120)
        ax.plot(fl.traceXYZ[-1,0],
                fl.traceXYZ[-1,1],
                fl.traceXYZ[-1,2],
                ms=2, marker='.',
                alpha=0.8, color='yellow', zorder=120)
        ax.plot(fl.traceXYZ[:,0],
                fl.traceXYZ[:,1],
                fl.traceXYZ[:,2],
                lw=1, ls='--', alpha=0.5, color='white', zorder=100)

    #  ax.axes.set_xlim3d(left=0.2, right=9.8)
    #  ax.axes.set_ylim3d(bottom=0.2, top=9.8)
    #  ax.axes.set_zlim3d(bottom=0.2, top=9.8)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)

    #  ax.set_box_aspect([1,1,1])
    figure_output(fig, ioconfig.plot)
    exit()

    plotSurfaceMappingMeridian(fls,
                               flattening=SIM.config["FLATTENING"],
                               config=(ioconfig.plot | ioconfig.figs.meridian_field_lines),
                               time=None)
    exit()
    # -------------------------------------

    # if config.time is not None:
    #     if config.name is not None:
    #         fullTitle = config.name + ' | ' + config.time
    #     else:
    #         fullTitle = config.time
    CLAT0 = 70
    CLAT1 = 85
    plotBins = LatBins
    moons = ['Enceladus', 'Rhea', 'Titan']
    moons = ['10', '20', '30']
  #  CLAT0 = 20
  #  CLAT1 = 70
  #  plotBins = MagLatBins
  #  moons = None

    if args.save:
      ioconfig.path = 'Output'
      ioconfig.save = True
      ioconfig.id = args.iter

    plotDwellMatrix = True
    plotEventMatrix = False
    plotRatioMatrix = False
    plotMagLat = False
  #  xlabel = 'Local Time'
    if plotDwellMatrix:
        xlabel = 'Local Time'
        ylabel = 'Conjugate Latitude'
        cbar_label = 'Cassini Dwell Time (days)'
        moons = None
        suptitle = None
    if plotRatioMatrix:
        xlabel = 'Local Time'
        cbar_label = 'Ratio of Event Time to Dwell Time'
        suptitle = 'Normalized Event Time for Total Dwell Time > %.1f' % minBound + ' days'
        moons = None
    if plotEventMatrix:
        moons = None
        xlabel = 'Local Time'
        ylabel = r'Conjugate Latitude ($^\circ$)'
        cbar_label = 'Cumulative Event Time (days)'
        suptitle = None
    if plotMagLat:
        ylabel = r'Magnetic Latitude'

    # plotOrbit3D(Data=R[:, 0 : truncateIndex], fls=fls, n=n, n_total=snaps, viewAngleChange=viewAngleChange, axisLim = 4, config=ioconfig)
    if args.movie:
        processMovie2(os.path.join(ioconfig.path, 'heatmap_%04d.png'),
                      res='5136x2880',
                      fps=30,
                      output=os.path.join(ioconfig.path, 'heatmap_movie.mp4'))
