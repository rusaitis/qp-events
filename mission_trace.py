import os
from sys import path
import copy
import argparse
import datetime
from datetime import timezone
import numpy as np
import pandas as pd
# Plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
# Local Plugins
import cassinilib
from cassinilib.DataPaths import *
from cassinilib.NewSignal import *
from cassinilib.Plot import *
from cassinilib.DatetimeFunctions import *
from wavesolver.sim import *
from wavesolver.fieldline import *
from wavesolver.io import *
from wavesolver.linalg import *
from wavesolver.model import *
from wavesolver.helperFunctions import *
import kmag

def loadMagnetosphereCrossings(filename=None):
    """ Load Magnetosphere Crossings (Jackman et al., 2019) """
    if filename is None:
        filename = '/Users/leo/DATA/CASSINI_DATA/DataProducts/CROSSINGS.npy'
    CROSSINGS = np.load(filename, allow_pickle=True)
    CDICT = {}
    print(CROSSINGS[0][0])
    print(CROSSINGS[0][-1])
    for year in arange(2004, 2018):
        dt0 = datetime.datetime.strptime(f'{year}', '%Y')
        dt1 = datetime.datetime.strptime(f'{year+1}', '%Y')
        ind0 = np.argwhere(CROSSINGS[0] >= dt0)[0][0]
        ind1 = np.argwhere(CROSSINGS[0] >= dt1)
        ind1 = CROSSINGS[0].size-1 if ind1.size == 0 else ind1[0][0]
        year_dict = {f'{year}': CROSSINGS[:3, ind0:ind1]}
        CDICT = CDICT | year_dict
    return CDICT

def location_in_magnetosphere(MP_crossings, time):
    """ Check if the Spacecraft is in the magnetosphere by Jackman+ (2015) """
    loc = 9
    year = time.strftime('%Y')
    data = MP_crossings.get(f'{year}')
    if data is not None:
        ind = np.argwhere(data[0] >= time)
        if ind.size > 0:
            loc = data[1][ind[0][0]]
    return loc

def configure_sweep(config, TIME):
    """ Configure the Orbital Sweep step parameters """
    config.data_size = np.size(TIME)
    config.date_from = TIME[0]
    config.date_to = TIME[-1]
    timeElapsed = (config.date_to - config.date_from).total_seconds() / 60
    config.total_steps = int(ceil(timeElapsed / config.res_in_min))
    config.index_step = (config.data_size + 1) // config.total_steps
    config.index_step = 1 if config.index_step == 0 else config.index_step
    config.total_steps = config.data_size // config.index_step

def check_data_validity(config, rotation_args=None):
    """ Check if the new data is valid for collection/analysis """
    if config.time_now == config.time_pre:
        print('Times between the points are equal.')
        return False
    if config.field_pre is None:
        print('No Previous Field Data Point found.')
        return False

    PHI = np.degrees(Position_RTP[2])
    LT = phi2LT(Position_RTP[2])

    # Magnetic Field Vectors and Angles
    Field_unit = Field / np.linalg.norm(Field)
    Field_last_unit = Field_last / np.linalg.norm(Field_last)
    Field_avg_unit = Field_avg / np.linalg.norm(Field_avg)
    Field_avg_last_unit = Field_avg_last / np.linalg.norm(Field_avg_last)
    BT = config.field_now[3]
    FieldAngle         = np.degrees(Lat2Colat(car2sph(Field)[1])) # wrt equator
    FieldAngle_last    = np.degrees(Lat2Colat(car2sph(Field_last)[1]))
    FieldAvgAngle      = np.degrees(Lat2Colat(car2sph(Field_avg)[1]))
    FieldAvgAngle_last = np.degrees(Lat2Colat(car2sph(Field_avg_last)[1]))
    dFieldAngle = FieldAngle - FieldAngle_last
    dFieldAvgAngle = FieldAvgAngle - FieldAvgAngle_last
    # Avg vector between the two times
    Field3 = (Field_avg + Field_avg_last)/2
    Field3Angle      = np.degrees(Lat2Colat(car2sph(Field3)[1])) # wrt equator
    Field3_unit = Field3 / np.linalg.norm(Field3)

    # Spacecraft Angles
    RS = 60268*1E3
    SC_vector = Position - Position_last
    SC_speed = np.linalg.norm(SC_vector)*RS/timeElapsed
    SC_unit_vector = SC_vector / np.linalg.norm(SC_vector)
    dFieldAngleNorm = dFieldAvgAngle / np.linalg.norm(SC_vector)

    # Spacecraft Angle with the Magnetic Field
    SC_field_angle = np.degrees(np.arccos(np.dot(SC_unit_vector, Field_unit)))
    SC_field_avg_angle = np.degrees(np.arccos(np.dot(SC_unit_vector, Field_avg_unit)))
    
    # Save for Output if needed
    config.position_now = Position
    config.position_pre = Position_last
    config.field_now = Field
    config.field_last = Field_last
    config.dBAngle = dFieldAngle
    config.dBavgAngle = dFieldAvgAngle
    config.BAngle = FieldAngle
    config.dBAngle_norm = dFieldAngleNorm
    config.BavgAngle = FieldAvgAngle
    config.SC_B_angle = SC_field_angle
    config.SC_Bavg_angle = SC_field_avg_angle
    config.SC_speed = SC_speed
    config.BT = BT
    config.LT = LT
    config.TH = TH
    config.PHI = PHI
    config.r = r
    config.loc = loc

    # Print Progress if Requested
    if config.n % 10000 == 0:
        print(config.n, '/', config.total_steps + 1, 'Trace',
              ' | DATA[', config.index_now, '/', config.data_size - 1, ']'
              ' |', config.time_now,
              ' | R:', config.position_now,
              ' | r: %.2f' % np.linalg.norm(np.array(config.position_now)))

    # Conditions for valid data
    field_angle_small_to_equator = np.abs(FieldAngle) <= 30
    field_strength_small = BT < 2
    within_distance = (r > 1.1 and r <= 80.)
    within_magnetosphere = loc == 0
    spacecraft_along_the_field = SC_field_angle < 10
    field_direction_change_is_small = dFieldAngle < 5

    if (within_distance
        #  and within_magnetosphere
        #  and field_angle_small_to_equator
        #  and field_strength_small
        #  and spacecraft_along_the_field
        #  and field_direction_change_is_small
        ):
        return True
    else:
        return False


if __name__ == "__main__":

    # ------------------------------------------------------------------------
    # Command line aguments
    parser = argparse.ArgumentParser(description='Process the waves.')
    parser.add_argument('--calc', dest='calculate', action='store_true',
                        help='Compute the Alfven waves')
    parser.add_argument('--mov', dest='movie',
                        action='store_true', help='Generate a movie')
    parser.add_argument('--save', dest='save',
                        action='store_true', help='Save the figures')
    parser.add_argument('--plot', dest='plot',
                        action='store_true', help='Show the Plot')
    parser.add_argument('--saveProgress', dest='saveProgress',
                        action='store_true', help='Save progress')
    parser.add_argument('--dipole', dest='dipole',
                        action='store_true', help='Use a dipole field')
    parser.add_argument('--spacecraftPosition', dest='spacecraftPosition',
                        action='store_true', help='Investigate Spc position')
    parser.add_argument('--fast', dest='fast', action='store_true',
                        help='Enable a faster calculation (less accurate)')
    parser.add_argument('--step', dest='step', type=float,
                        default=0.1,
                        help='Field line tracing step size (in planet radii)')
    parser.add_argument('--f', dest='f', type=float,
                        default=0.09796,
                        help="Flatenning of the Planet's surface")
    parser.add_argument('--maxR', dest='maxR', type=float,
                        default=100.,
                        help='Maximum allowed distance of a field line point')
    parser.add_argument('--output', dest='output', type=str,
                        help='Custom output folder')
    parser.add_argument('--method', dest='method', type=str, default='euler',
                        help='Trace method: Euler or RK4')
    parser.add_argument('--year', dest='year', type=int, default=2005,
                        help='Year to analyze the data')
    args = parser.parse_args()
    # ------------------------------------------------------------------------

    # Set up the output folder and other plotting parameters
    if args.output is not None:
        customFolder = args.output
        outputPath = os.path.join('Output', customFolder)
        createDirectory(outputPath)
    else:
        outputPath = 'Output'

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

    # Delete old *.png files in the output directory
    # deleteFiles(ioconfig.path, fileformat='.pdf')

    MP_crossings = loadMagnetosphereCrossings(filename=None) # Jackman+ (2019)

    # Load Saturn configuration (browse configurations.py for more options)
    SIM = loadsim(configSaturnNominal)
    SIM.config["ETIME"] = 1483228900  # (01 Jan 2017)
    SIM.config["FLATTENING"] = 1. # Default is 0.9794
    SIM.config["IN_COORD"] = 'KSM'  # S3C/DIS/DIP/KSM/KSO
    SIM.config["OUT_COORD"] = 'DIS'  # S3C/DIS/DIP/KSM/KSO
    SIM.config["IN_CARSPH"] = 'CAR'  # INPUT IN SPH/CAR
    SIM.config["OUT_CARSPH"] = 'CAR'  # OUTPUT IN SPH/CAR
    SIM.config["step"] = float(args.step)
    SIM.config["maxIter"] = 1E6
    SIM.config["method"] = args.method
    SIM.config["maxR"] = args.maxR
    SIM.step = float(args.step)

    EVENTS = np.loadtxt(os.path.join(ioconfig.data_dir,
                                     'cassini-events',
                                     'events_proposal.dat'),
                        skiprows=0,
                        dtype='str')
#    EVENTS = EVENTS[0:3]
    EVENTS_FROM = EVENTS[:,0]
    EVENTS_TO = EVENTS[:,1]
    EVENTS_FROM = [datetime.datetime.strptime(x, DATETIME_FORMAT[0])
                   for x in EVENTS_FROM]
    EVENTS_TO = [datetime.datetime.strptime(x, DATETIME_FORMAT[0])
                 for x in EVENTS_TO]

    if args.year == 2004:
        dateFrom = '2004-06-30T00:00:00'
        dateTo = '2005-01-01T00:00:00'
    else:
        dateFrom = str(args.year) + '-01-01T00:00:00'
        dateTo = str(args.year+1) + '-01-01T00:00:00'

    # Grand Finale
    #  dateFrom = '2017-04-01T00:00:00'
    #  dateTo = '2017-09-15T00:00:00'

    # Southern Solstace
    #  dateFrom = '2004-06-30T00:00:00'
    #  dateTo = '2009-06-01T00:00:00'

    # Northern Solstace
    #  dateFrom = '2009-06-01T00:00:00'
    #  dateTo = '2017-09-15T00:00:00'

    dateFrom = '2004-06-30T00:00:00'
    #  dateTo = '2008-01-30T00:00:00'
    dateTo = '2017-09-15T00:00:00'

    sweep_config = {"res_in_min": 30,
                    "avg_ind_steps": 6,
                    "data": 'source',
                    "mapping_mode": 'spacecraft',
                    "load_data": True,
                    "trace": True,
                    "save_every_step": 40,
                    "save_progress": False,
                    "data_request_from": dateFrom,
                    "data_request_to": dateTo,
                    "data_from": 0,
                    "data_to": 0,
                    "n": 0,
                    "index_pre": 0,
                    "index_now": 1,
                    "time_pre": 0,
                    "time_now": 0,
                    "field_pre": None,
                    "field_now": None,
                    "position_pre": None,
                    "position_now": None,
                    "data_size": 0,
                    "total_steps": 0,
                    "log_bad_intervals": [],
                    "MP_crossings": None,
                    "BAngle": None,
                    "dBAngle_norm": None,
                    "BavgAngle": None,
                    "dBAngle": None,
                    "dBavgAngle": None,
                    "SC_B_angle": None,
                    "SC_Bavg_angle": None,
                    "SC_speed": None,
                    "continuous": True,
                    "BT": None,
                    "LT": None,
                    "TH": None,
                    "PHI": None,
                    "r": None,
                    "loc": None,
                    }

    sweep = Config(sweep_config)
    sweep.MP_crossings = MP_crossings

    if sweep.data == 'source':
        print(f'Dates: {sweep.data_request_from} - {sweep.data_request_to}')
        timeseries = readSignal(sweep.data_request_from,
                                sweep.data_request_to,
                                instrument = 'MAG',
                                measurement = '1min',
                                coord = ioconfig.coord,
                                file = None,
                                marginInSec = 0,
                                dt = sweep.res_in_min * 60,
                                datetimeFormat=DATETIME_FORMAT[0])
        #  R = np.load(os.path.join(ioconfig.data_dir, 'cassini-events', 'events.npy'))
        #  Time = np.load(os.path.join(ioconfig.data_dir, 'cassini-events', 'events_time.npy'),
                       #  allow_pickle=True)
            #  datafile = 'Cassini_MAG_KSM_24H.npy'
            #  fullpath = os.path.join(OUTPUT_DIR, datafile)
            #  CROSSINGS_FILE = os.path.join(OUTPUT_DIR, 'CROSSINGS.npy')
            #  CROSSINGS = np.load(CROSSINGS_FILE, allow_pickle=True)

#     Time, R = simulateOrbitData(dateFrom='2005-02-01T00:00:00',
#                                 nphi=30, rstart=20.27) # Titan
#    orbitPhi = [phi for phi in np.linspace(0, 2. * np.pi, 10)]
        # TIMESERIES FOR THE MOONS
        # timeseries = readSignal(dateFrom,
        #                        dateTo,
        #                        instrument = 'SIM',
        #                        measurement = 'Titan',
        #                        coords = 'KSM',
        #                        resolutionInMin = 60*4,
        #                        datetimeFormat = DATETIME_FORMAT_ORBITS,
        #                        dt = 1)
        TIME = [s.y for s in timeseries if s.kind == 'Time'][0]
        coords = [s for s in timeseries if s.kind == 'Coord']
        fields = [s for s in timeseries if s.kind == 'Field']
        FIELD = np.stack([list(coord) for coord in fields], axis=0)
        R = np.stack([list(coord) for coord in coords], axis=0)

    configure_sweep(sweep, TIME)
    print('size of R:', shape(R))
    print('size of FIELD:', shape(FIELD))
    orbit_data = {"r": [], "TH": [], "PHI": [], "LT": [], "BT": [], "loc": [],
                  "BAngle": [], "dBAngle": [], "SC_speed": [], "SC_B_angle":[],
                  "BavgAngle": [], "dBavgAngle": [], "SC_Bavg_angle": [],
                  "continuous": [], "dBAngle_norm": []}

    if coords[1].units in ['deg', 'rad']:
        print('Detected Spherical Coords. Adjusting.')
        dataInSphericalCoords = True
        R = cassinilib.sph2car(R)

    if TIME:
        print('Data Loaded. Starting to sweep . . .')

        LatBins = ioconfig.nbins.Lat
        LTBins = ioconfig.nbins.LT
        MlatBins = ioconfig.nbins.Mlat
        RBins = ioconfig.nbins.R

        SurfaceMap       = np.zeros((LatBins, LTBins, MlatBins))
        SurfaceMapOpen   = np.zeros((LatBins, LTBins, MlatBins))
        SurfaceMapClosed = np.zeros((LatBins, LTBins, MlatBins))
        SurfaceMapNearby = np.zeros((LatBins, LTBins, MlatBins))
        RMap = np.zeros(RBins)
        continueTracing = True

        while continueTracing:
            sweep.n += 1
            sweep.time_now = TIME[sweep.index_now]
            sweep.time_pre = TIME[sweep.index_pre]
            timeElapsed = (sweep.time_now - sweep.time_pre).total_seconds()
            #  sweep.time_now += datetime.timedelta(days=360*6)
            #  SIM.config["IN_COORD"] = SIM.config["OUT_COORD"]
            SIM.config["ETIME"] = date2timestamp(sweep.time_now)
            rotation_args = lambda vec: [SIM.config["IN_COORD"],
                SIM.config["OUT_COORD"], vec, SIM.config["ETIME"], SIM.config["EPOCH"]]
            sweep.position_now = R[:, sweep.index_now]
            sweep.position_pre = R[:, sweep.index_pre]
            sweep.field_now = FIELD[:, sweep.index_now]
            sweep.field_pre = FIELD[:, sweep.index_pre]
            #  print(f'Sweep Index 1: {sweep.index_now}')
            #  print(f'Sweep Index 0: {sweep.index_pre}')
            nn = sweep.avg_ind_steps // 2
            AVG_ind_from = [sweep.index_pre - nn, sweep.index_now - nn]
            AVG_ind_to = [sweep.index_pre + nn, sweep.index_now + nn]
            AVG_ind_from = [0 if x < 0 else x for x in AVG_ind_from]
            AVG_ind_to = [sweep.data_size-1 if x > sweep.data_size-1 else x for x in AVG_ind_to]
            field_avg_pre = np.average(FIELD[:3, AVG_ind_from[0]:AVG_ind_to[0]], axis=-1)
            field_avg_now = np.average(FIELD[:3, AVG_ind_from[1]:AVG_ind_to[1]], axis=-1)
            dt_avg0 = (TIME[AVG_ind_to[0]] - TIME[AVG_ind_from[0]]).total_seconds() / 60
            dt_avg1 = (TIME[AVG_ind_to[1]] - TIME[AVG_ind_from[1]]).total_seconds() / 60
            if (dt_avg0 > 2 * nn * sweep.res_in_min) or (dt_avg1 > 2 * nn * sweep.res_in_min):
                sweep.continuous = False
            else:
                sweep.continuous = True

            #  print(field_avg_pre)
            #  print(field_avg_now)
            #  print(np.average(field_avg_pre, axis=-1))
            #  print(np.average(field_avg_now, axis=-1))
            sweep.field_avg_pre = field_avg_pre
            sweep.field_avg_now = field_avg_now

            if check_data_validity(sweep, rotation_args=rotation_args):
                orbit_data["r"].append(sweep.r)
                orbit_data["TH"].append(sweep.TH)
                orbit_data["PHI"].append(sweep.PHI)
                orbit_data["LT"].append(sweep.LT)
                orbit_data["BT"].append(sweep.BT)
                orbit_data["BAngle"].append(sweep.BAngle)
                orbit_data["BavgAngle"].append(sweep.BavgAngle)
                orbit_data["dBAngle"].append(sweep.dBAngle)
                orbit_data["dBavgAngle"].append(sweep.dBavgAngle)
                orbit_data["dBAngle_norm"].append(sweep.dBAngle_norm)
                orbit_data["SC_speed"].append(sweep.SC_speed)
                orbit_data["SC_B_angle"].append(sweep.SC_B_angle)
                orbit_data["SC_Bavg_angle"].append(sweep.SC_Bavg_angle)
                orbit_data["loc"].append(sweep.loc)
                orbit_data["continuous"].append(sweep.continuous)

                if sweep.mapping_mode == 'dipole':
                    R1,R2,TH1,TH2,LT1,LT2 = dipFieldMap(Position,SIM)
                    LT1 = phi2LT(LT1)
                    LT2 = phi2LT(LT2)
                    R0 = car2sph(Position)
                    TH0 = R0[1]
                elif sweep.mapping_mode == 'spacecraft':
                    R1, R2        = [sweep.r] * 2
                    TH1, TH2, TH0 = [Lat2Colat(np.radians(sweep.TH))] * 3
                    LT1, LT2      = [sweep.LT] * 2
                    R0 = [sweep.r, Lat2Colat(np.radians(sweep.TH)), np.radians(sweep.PHI)]
                    if sweep.TH > 0:
                        R1, R2 = 1.  , 200.
                    else:
                        R1, R2 = 200., 1.
                elif sweep.mapping_mode == 'kmag':
                    fl = traceFieldline(sweep.position_now, SIM)
                    R1  = fl.traceRTP[ 0, 0]
                    R2  = fl.traceRTP[-1, 0]
                    TH1 = fl.traceRTP[ 0, 1]
                    TH2 = fl.traceRTP[-1, 1]
                    LT1 = phi2LT(fl.traceRTP[ 0, 2])
                    LT2 = phi2LT(fl.traceRTP[-1, 2])
                    R0 = car2sph(fl.traceStart)
                    TH0 = R0[1]

                i_TH1 = value2bin(Colat2Lat(TH1), -np.pi/2, np.pi/2, LatBins)
                j_LT1 = value2bin(LT1, 0, 24, LTBins)
                i_TH2 = value2bin(Colat2Lat(TH2), -np.pi/2, np.pi/2, LatBins)
                j_LT2 = value2bin(LT2, 0, 24, LTBins)
                k_TH0 = value2bin(Colat2Lat(TH0), -np.pi/2, np.pi/2, MlatBins)

                ReachedNorthSurface = R1 <= 1.5
                ReachedSouthSurface = R2 <= 1.5

                # Bins start South to North in colatitude, 0 to 24 in LT
                if ReachedNorthSurface:
                    SurfaceMap[i_TH1, j_LT1, k_TH0] += timeElapsed
                    if sweep.r < 30:
                        SurfaceMapNearby[i_TH1, j_LT1, k_TH0] += timeElapsed

                if ReachedSouthSurface:
                    SurfaceMap[i_TH2, j_LT2, k_TH0] += timeElapsed
                    if sweep.r < 30:
                        SurfaceMapNearby[i_TH2, j_LT2, k_TH0] += timeElapsed

                if ReachedNorthSurface and ReachedSouthSurface:
                    SurfaceMapClosed[i_TH1, j_LT1, k_TH0] += timeElapsed
                    SurfaceMapClosed[i_TH2, j_LT2, k_TH0] += timeElapsed

                if ReachedNorthSurface and not ReachedSouthSurface:
                    SurfaceMapOpen[i_TH1, j_LT1, k_TH0] += timeElapsed

                if ReachedSouthSurface and not ReachedNorthSurface:
                    SurfaceMapOpen[i_TH2, j_LT2, k_TH0] += timeElapsed

                if sweep.r < 50:
                    RMap[value2bin(sweep.r, 0., 50., RBins)] += timeElapsed

            sweep.index_pre = sweep.index_now

            continueTracing = sweep.index_now < (sweep.data_size - 1)
            sweep.index_now += sweep.index_step

            if args.saveProgress:
                if n % saveFrequency == 0:
                    dstring_from = sweep.date_from.strftime(DATE_SHORT)
                    dstring_to = sweep.time_now.strftime(DATE_SHORT)
                    f_ = ioconfig.plot.folder
                    fstr = lambda s: (f'{s}_{saveprogress:05d}_{dstring_from}'
                                      f'_to_{dstring_to}.npy')
                    np.save(os.path.join(f_, fstr('surfaceMap_')), SurfaceMap)
                    np.save(os.path.join(f_, fstr('fieldline_')), fl)
                    np.save(os.path.join(f_, fstr('R_')), RMap)
                    saveProgress += 1

        dstring_from = sweep.date_from.strftime(DATE_SHORT)
        dstring_to = sweep.date_to.strftime(DATE_SHORT)
        fstr = lambda s: f'{s}_Total_{dstring_from}_to_{dstring_to}.npy'
        f_ = ioconfig.plot.folder
        np.save(os.path.join(f_, fstr('surfaceMap')), SurfaceMap)
        np.save(os.path.join(f_, fstr('surfaceMapClosed')), SurfaceMapClosed)
        np.save(os.path.join(f_, fstr('surfaceMapOpen')), SurfaceMapOpen)
        np.save(os.path.join(f_, fstr('surfaceMapNearby')), SurfaceMapNearby)
        np.save(os.path.join(f_, fstr('R_')), RMap)

        if size(sweep.log_bad_intervals) > 0:
            bad = [[bad[i].strftime(DATETIME_FORMAT[0]) for i in [0,1]]
                   for bad in sweep.log_bad_intervals]
            np.savetxt(os.path.join(ioconfig.plot.folder,
                                    ('dataDiscontinuities_' +
                                     sweep.date_from.strftime(DATE_SHORT) +
                                     '_to_' +
                                     sweep.date_to.strftime(DATE_SHORT) +
                                     '.out')),
                       bad, fmt='%20s')


        def classify_r(r):
            if r > 30:
                return "r>30"
            elif r > 20:
                return "r=20-30"
            elif r > 10:
                return "r=10-20"
            else:
                return 'r=0-10'

        orbit_data = pd.DataFrame.from_dict(orbit_data)
        orbit_data = orbit_data[orbit_data["loc"] == 0]
        orbit_data = orbit_data[orbit_data["continuous"] == True]
        orbit_data = orbit_data[orbit_data["r"] < 50]
        #  orbit_data = orbit_data[np.abs(orbit_data["BavgAngle"]) < 20]
        #  orbit_data = orbit_data[orbit_data["r"] > 3]
        orbit_data = orbit_data.assign(BT_cat=lambda x: x.BT < 4)
        orbit_data = orbit_data.assign(r_cat=lambda x: [classify_r(a) for a in x.r])
        orbit_data = orbit_data.assign(SC_B_angle_min=lambda x: np.min([x.SC_Bavg_angle, 180-x.SC_Bavg_angle], axis=0))
        orbit_data = orbit_data[orbit_data["SC_B_angle_min"] < 30]
        orbit_data = orbit_data[orbit_data["dBAngle_norm"] < 5]

        #  orbit_data = orbit_data[orbit_data["r_cat"] == "r=10-20"]
        #  orbit_data = orbit_data.assign(r_cat=lambda x: x.r < 20)
        #  orbit_data.assign(r_cat=[classify_r(a) for a in orbit_data.r])

        mpl.rcParams['figure.figsize'] = (6,4)
        mpl.rcParams['figure.dpi'] = 150
        mpl.rcParams['savefig.dpi'] = 150
        import seaborn as sns
        #  sns.set_theme()
        sns.set_theme(style="darkgrid")
        sns.set_context("paper")
        sns.set_style("ticks", {"axes.grid": True, "grid.color": ".8", "grid.linestyle": ":"})
        # sns.set_theme(style="ticks", palette="deep")
        #  sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

        #  fig = plt.figure(figsize=(6,4), dpi=150)
        #  ax = fig.add_subplot(111)

        #  g = sns.jointplot(data=orbit_data, x="LT", y="TH",
                          #  kind="hist", hue="loc",
                          #  #  truncate=False,
                          #  xlim=(0, 24), ylim=(-90, 90),
                          #  palette='deep',
                          #  color="m",
                          #  height=5, ratio=2, marginal_ticks=True,
                          #  ax=ax)

        #  sns.set(rc={'figure.figsize':(8,4)})
        g = sns.JointGrid(data=orbit_data, x="LT", y="TH",
                          xlim=(0, 24),
                          ylim=(-90, 90),
                          hue="r_cat", palette='Set2',
                          marginal_ticks=True,
                          #  height=6
                          )
        #  plt.legend(labels=["Magnetosphere", "Magnetosheath", "Solar Wind", "Uncategorized"], title = "Cassini Location")
        g.figure.set_size_inches(6,4)
        #  fig.set_size_inches(18.5, 10.5, forward=True)

        g.plot_joint(sns.scatterplot, s=2, alpha=.25)
        g.plot_marginals(sns.histplot, kde=True, bins=50)
        #  g.refline(x=12, y=0)

        g.set_axis_labels(r"Local Time [h]", r"Latitude [deg]")
        g.set_axis_labels(r"Local Time [h]", r"Magnetic Field Angle to Mag Equator [deg]")
        #  g.set_axis_labels(r"Local Time [h]", r"Magnetic Field Angle to Spacecraft Trajectory [deg]")
        #  g.set_axis_labels(r"Latitude [deg]", r"Magnetic Field Angle to Spacecraft Trajectory [deg]")
        #  leg = g.axes.flat[0].get_legend()
        leg = g.figure.get_axes()[0].get_legend()
        #  leg = g.figure.get_legend()
        #  g.figure.get_axes()[0].set(xscale="log")
        leg.set_title('Location')
        #  new_labels = ["Magnetosphere", "Magnetosheath", "Solar Wind"]
        #  new_labels = ["Magnetosphere"]
        #  for t, l in zip(leg.texts, new_labels):
            #  t.set_text(l)
        g.figure.get_axes()[0].set_xticks(np.arange(0, 24+3, 3))

        #  fig.savefig('test2png.png', dpi=100)
        #  plt.legend(title='Smoker', loc='upper left', labels=['Hell Yeh', 'Nah Bruh'])
        g.figure.tight_layout()
        plt.show()
        exit()

#
        #  g = sns.JointGrid()
        #  x, y = data_orbit["LT"], data_orbit["TH"]
        #  sns.scatterplot(x=x, y=y, s=10, ax=g.ax_joint)
        #  sns.histplot(x=x, linewidth=2, ax=g.ax_marg_x)
        #  sns.kdeplot(y=y, linewidth=2, ax=g.ax_marg_y)

        #  g = sns.JointGrid(data=orbit_data, x="LT", y="TH", hue="loc", size="BT_cat")
        #  g.plot(sns.scatterplot, sns.histplot)

        #  df = np.random.rand(10, 12)
        #  ax = sns.heatmap(df)

        #  g = sns.JointGrid(data=orbit_data, x="LT", y="TH", space=0)
        #  g = sns.JointGrid()
        #  g.plot_joint(sns.heatmap,
                     #  data=df,
                     #  #  fill=True,
                     #  #  clip=((2200, 6800), (10, 25)),
                     #  #  thresh=0, levels=100, cmap="rocket",
                     #  )
        #  g.plot_marginals(sns.histplot, color="#03051A", alpha=1, bins=25)


        #  g = sns.jointplot(data=df, kind='hist', bins=(10,12))
        #  g.ax_marg_y.cla()
        #  g.ax_marg_x.cla()
        #  sns.heatmap(data=df, ax=g.ax_joint, cbar=False, cmap='Blues')
#

        #  g = sns.scatterplot(data=orbit_data,
                            #  x="LT",
                            #  y="TH",
                            #  hue="loc",palette="deep",
                            #  s=3,
                            #  marker="x",
                            #  ax=ax)

        #  g = sns.histplot(
                #  orbit_data, x="LT", y="TH", hue="loc",
                #  bins=100, discrete=(True, False), log_scale=(False, False),
                #  cbar=True, cbar_kws=dict(shrink=.75),
            #  )

        #  g.set_axis_labels("Flipper length (mm)", "Bill length (mm)")

        #  #  fig.patch.set_alpha(0.)
        #  #  ax.patch.set_alpha(0.)
        #  #  ax.scatter(Lats, BAngles, alpha=0.02)
        #  #  ax.scatter(orbit_data["LT"], orbit_data["TH"], s=1, alpha=0.2)
        #  x =  orbit_data["LT"]
        #  y =  orbit_data["TH"]
        #  #  y =  orbit_data["BAngle"]
        #  #  y =  orbit_data["dBAngle"]
        #  #  y =  orbit_data["SC_B_angle"]
        #  r =  orbit_data["r"]
        #  #  ax.scatter(x[np.where((th<10))], y[np.where((th<10))], s=1, alpha=0.1)
        #  ax.scatter(x, y, s=1, alpha=0.1)
        #  ax.axhline(y=0, ls='--', alpha=0.4, color='black')
        #  #  ax.axvline(x=0, ls='--', alpha=0.4, color='black')
        #  ax.set_xlim(0,24)
        #  #  ax.set_xlim(-90, 90)
        #  ax.set_ylim(-90, 90)
        #  #  ax.set_ylim(-10, 10)
        #  ax.grid()
        #  #  ax.set_ylim(-90, 90)
        #  plt.savefig('Output/figure.png')
        #  plt.close()
        #  plt.show()
        exit()

        if args.plot:

            LatBins = len(SurfaceMap[:,0,0])
            LTBins = len(SurfaceMap[0,:,0])
            MagLatBins = len(SurfaceMap[0,0,:])
            LatBins = int(LatBins)
            LTBins = int(LTBins)
            MagLatBins = int(MagLatBins)

            ss = 0
            plotHeatmap(np.sum(SurfaceMap / (60 * 60 * 24), axis=2),
                        xRange = [0, 24],
                        yRange = [-90, 90],
                        splitRange = [[ ss,  90],
                                      [-90, -ss]],
                        splitBins=[[value2bin( ss, minValue=-90, maxValue=90,
                                              bins=LatBins, range=True),
                                    value2bin( 90, minValue=-90, maxValue=90,
                                              bins=LatBins, range=True)],
                                   [value2bin(-90, minValue=-90, maxValue=90,
                                              bins=LatBins, range=True),
                                    value2bin(-ss, minValue=-90, maxValue=90,
                                              bins=LatBins, range=True)]],
                        margin=value2bin( 2, minValue=0,
                                         maxValue=24, bins=LTBins),
                        #  moons = ['10', '20'],
                        config=Config(ioconfig.plot | ioconfig.figs.heatmap),
                        #  config=ioconfig,
                        )
            # ss = 60
            # plotHeatmapPolar(np.sum(SurfaceMap / (60 * 60 * 24), axis=2),
            #               xRange = [0, 24],
            #               yRange = [-90, 90],
            #               splitRange = [[ ss,  90],
            #                             [-90, -ss]],
        #  #  ax.scatter(x[np.where((th<10))], y[np.where((th<10))], s=1, alpha=0.1)
        #  ax.scatter(x, y, s=1, alpha=0.1)
        #  ax.axhline(y=0, ls='--', alpha=0.4, color='black')
        #  #  ax.axvline(x=0, ls='--', alpha=0.4, color='black')
        #  ax.set_xlim(0,24)
        #  #  ax.set_xlim(-90, 90)
        #  ax.set_ylim(-90, 90)
        #  #  ax.set_ylim(-10, 10)
        #  ax.grid()
        #  #  ax.set_ylim(-90, 90)
        #  plt.savefig('Output/figure.png')
        #  plt.close()
        #  plt.show()
        exit()

        if args.plot:

            LatBins = len(SurfaceMap[:,0,0])
            LTBins = len(SurfaceMap[0,:,0])
            MagLatBins = len(SurfaceMap[0,0,:])
            LatBins = int(LatBins)
            LTBins = int(LTBins)
            MagLatBins = int(MagLatBins)

            ss = 0
            plotHeatmap(np.sum(SurfaceMap / (60 * 60 * 24), axis=2),
                        xRange = [0, 24],
                        yRange = [-90, 90],
                        splitRange = [[ ss,  90],
                                      [-90, -ss]],
                        splitBins=[[value2bin( ss, minValue=-90, maxValue=90,
                                              bins=LatBins, range=True),
                                    value2bin( 90, minValue=-90, maxValue=90,
                                              bins=LatBins, range=True)],
                                   [value2bin(-90, minValue=-90, maxValue=90,
                                              bins=LatBins, range=True),
                                    value2bin(-ss, minValue=-90, maxValue=90,
                                              bins=LatBins, range=True)]],
                        margin=value2bin( 2, minValue=0,
                                         maxValue=24, bins=LTBins),
                        #  moons = ['10', '20'],
                        config=Config(ioconfig.plot | ioconfig.figs.heatmap),
                        #  config=ioconfig,
                        )
            # ss = 60
            # plotHeatmapPolar(np.sum(SurfaceMap / (60 * 60 * 24), axis=2),
            #               xRange = [0, 24],
            #               yRange = [-90, 90],
            #               splitRange = [[ ss,  90],
            #                             [-90, -ss]],
            #               splitBins=[[value2bin(-90, minValue=-90,
            #                          maxValue=90, bins=LatBins, range=True),
            #                           value2bin(-ss, minValue=-90,r
            #                         maxValue=90, bins=LatBins, range=True)],
            #                          [value2bin( ss, minValue=-90,
            #                         maxValue=90, bins=LatBins, range=True),
            #                           value2bin( 90, minValue=-90,
            #                        maxValue=90, bins=LatBins, range=True)]],
            #               margin=value2bin( 2, minValue=0,
            #                                maxValue=24, bins=LTBins),
            #               config=ioconfig)
    if args.movie:
        processMovie2(os.path.join(ioconfig.path, 'heatmap_%03d.png'),
                      res='5136x2880',
                      fps=30,
                  output=os.path.join(ioconfig.path, 'heatmap_movie.mp4'))
