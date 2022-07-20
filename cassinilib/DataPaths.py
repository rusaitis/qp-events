from cassinilib.Dataframe import Dataframe
import numpy as np
import os

# DATA_DIR = '/home/leo/DATA/CASSINI_DATA/' # UBUNTU
# DATA_DIR = '~/DATA/CASSINI_DATA/' # MAC
DATA_DIR = '/Users/leo/DATA/CASSINI_DATA/'
OUTPUT_DIR = '/Users/leo/DATA/CASSINI_DATA/DataProducts/'

EXPORT_FOLDER = 'PLOTS'

SCAS_FILE = '/Users/leo/DATA/CASSINI_DATA/DataProducts/SCAS_TIMES.npy'

COORDS = 'KRTP' # ['KRTP', 'KSM', 'KSO', 'RTN']
DATETIME_FORMAT = ['%Y-%m-%dT%H:%M:%S', '%Y-%jT%H:%M:%S', '%Y-%jT%H:%M:%S']
DATETIME_FORMAT_ORBITS = '%Y-%m-%dT%H:%M:%S.000Z'
DATE_SHORT = '%Y-%m-%d'
#  eventFile = 'events.dat'
#  eventFile = 'events_select.dat'
#  eventFile = 'events_sw.dat'
eventFile = 'events_proposal.dat'
# eventFile = 'events_scas.dat'

cred = '#fc4267'
cblue = '#0acaff'
cgreen = '#0aff9d'
clist_field = ['#DC267F', '#FFB000', '#FE6100', '#648FFF', '#785EF0', '#cf9bff', '#fdf33c',  '#ff396a']

clist_field = ['#fc4267', '#0acaff', '#0aff9d', 'white']
clist_field = ['#DC267F', '#FFB000', '#FE6100', '#648FFF', '#785EF0', '#cf9bff',
               '#fdf33c',  '#ff396a', 'red', 'green', 'blue', 'black', 'white',
               '#fc4267', '#0acaff', '#0aff9d', 'white', 'cyan', 'grey', 'yellow']

events_selected = ['2007-02-01T00:00:00', '2007-02-02T00:00:00',
        '2007-02-03T00:00:00', '2007-02-04T00:00:00', '2007-02-05T00:00:00',
        '2007-02-06T00:00:00', '2007-02-07T00:00:00', '2007-02-08T00:00:00',
        '2007-02-09T00:00:00', '2007-02-10T00:00:00', '2007-02-11T00:00:00',
        '2007-02-12T00:00:00', '2007-02-13T00:00:00', '2007-02-14T00:00:00',
        '2007-02-15T00:00:00', '2007-02-16T00:00:00', '2007-02-17T00:00:00',
        '2007-02-18T00:00:00', '2007-02-19T00:00:00', '2007-02-20T00:00:00',
        '2007-02-21T00:00:00', '2007-02-22T00:00:00', '2007-02-23T00:00:00',
        '2007-02-24T00:00:00', '2006-09-27T00:00:00', '2006-11-10T00:00:00',
        '2006-11-23T00:00:00', '2006-12-06T00:00:00', '2006-12-18T00:00:00',
        '2006-12-21T00:00:00', '2006-12-22T00:00:00', '2007-02-08T00:00:00',
        '2008-02-29T00:00:00', '2008-08-10T00:00:00', '2008-09-09T00:00:00',
        '2009-02-23T00:00:00', '2012-10-18T00:00:00', '2013-08-16T00:00:00']


# Cassini Time at Saturn
dateMissionFrom = '2004-06-30T00:00:00'
#  dateMissionTo = '2004-07-15T00:00:00'
dateMissionTo = '2017-09-13T00:00:00'

# Testing Times for Synthetic Data
test_dateMissionFrom = '2000-01-01T00:00:00'
test_dateMissionTo = '2000-01-10T00:00:00'
testFull_dateMissionFrom = '2000-01-01T00:00:00'
testFull_dateMissionTo = '2000-03-15T00:00:00'

SLS5_datafile = 'SLS5_2004-2018.txt'
BS_MP_crossing_datafile = 'BS_MP_Crossing.txt'

def dataFileName(instrument='MAG',
                 coords='KRTP',
                 measurement='1min',
                 dataDirectory=''):
    ''' Generate the file name for the wanted data source '''
    DATA_SOURCE_FILE_FORMAT = '.TAB'
    if instrument == 'MAG':
        DATA_SOURCE_FILE_START = ''
        if measurement == '1min':
            DATA_SOURCE_NAME = 'CO-E_SW_J_S-MAG-4-SUMM-1MINAVG-V1-2'
            DATA_SOURCE_NAME = 'CO-E_SW_J_S-MAG-4-SUMM-1MINAVG-V2'
            dataFilePath = os.path.join(dataDirectory,
                                        DATA_SOURCE_NAME,
                                        "DATA",
                                        "YYYY",
                                        ('YYYY' + '_FGM_'+ coords + '_1M'
                                         + DATA_SOURCE_FILE_FORMAT))
        if measurement == '1sec':
            DATA_SOURCE_NAME = 'CO-E_SW_J_S-MAG-4-SUMM-1SECAVG-V2'
            dataFilePath = os.path.join(dataDirectory,
                                        DATA_SOURCE_NAME,
                                        "DATA",
                                        "YYYY",
                                        ('YYDDD_YYDDD_XXX' + '_FGM_'+ coords
                                        + '_1S' + DATA_SOURCE_FILE_FORMAT))
    if instrument == 'CAPS':
        if measurement == 'ions':
            DATA_SOURCE_NAME = 'CO-S_SW-CAPS-5-DDR-ION-MOMENTS-V2'
            dataFilePath = os.path.join(dataDirectory,
                                        DATA_SOURCE_NAME,
                                        "DATA",
                                        "YYYY",
                                        ('ION_MOMT_' + 'YYYY' + '_01'
                                         + DATA_SOURCE_FILE_FORMAT))
        if measurement == 'electrons':
            DATA_SOURCE_NAME = 'CO-E_J_S_SW-CAPS-5-DDR-ELE-MOMENTS-V2'
            dataFilePath = os.path.join(dataDirectory,
                                        DATA_SOURCE_NAME,
                                        "DATA",
                                        "YYYY",
                                        ('ELS_3DMOMT_' + 'YYYY' + '_00'
                                         + DATA_SOURCE_FILE_FORMAT))
    if instrument == 'SIM':
        DATA_SOURCE_FILE_FORMAT = '.TAB'
        if measurement == '1min':
            DATA_SOURCE_NAME = 'SIM_1MINAVG'
            dataFilePath = os.path.join(dataDirectory,
                                        DATA_SOURCE_NAME,
                                        "YYYY",
                                        ('YYYY' + '_'+ coords + '_1M'
                                         + DATA_SOURCE_FILE_FORMAT))
        if measurement == 'Titan':
            DATA_SOURCE_NAME = 'ORBITS'
            DATA_SOURCE_FILE_FORMAT = '.txt'
            dataFilePath = os.path.join(dataDirectory,
                                        DATA_SOURCE_NAME,
                                        ('titan-' + 'YYYY' + '-' + coords
                                         + '-traj' + DATA_SOURCE_FILE_FORMAT))
    return dataFilePath

DataFields = {}
MAGDataFields = {}
SIMDataFields = {}

# MAG KRTP
MAGDataframe = []
# Standard colors: blue, green, red, black
MAGDataframe.append(Dataframe(0, 'Time', 'min', dt=60.0, type='str',   color='black', kind='Time', format='%Y-%m-%dT%H:%M:%S'))
MAGDataframe.append(Dataframe(1, 'Br',   'nT',  dt=60.0, type='float', color=clist_field[0], kind='Field'))
MAGDataframe.append(Dataframe(2, 'Bth',  'nT',  dt=60.0, type='float', color=clist_field[1], kind='Field'))
MAGDataframe.append(Dataframe(3, 'Bphi', 'nT',  dt=60.0, type='float', color=clist_field[2], kind='Field'))
MAGDataframe.append(Dataframe(4, 'Btot', 'nT',  dt=60.0, type='float', color=clist_field[3], kind='Field'))
MAGDataframe.append(Dataframe(5, 'r',    'R_S', dt=60.0, type='float', color='black', kind='Coord'))
MAGDataframe.append(Dataframe(6, 'th',   'deg', dt=60.0, type='float', color='black', kind='Coord'))
MAGDataframe.append(Dataframe(7, 'phi',  'deg', dt=60.0, type='float', color='black', kind='Coord'))
MAGDataframe.append(Dataframe(8, 'LT',   'h',   dt=60.0, type='float', color='black', kind='lt'))
MAGDataFields['KRTP'] = MAGDataframe

# MAG KSM/KSO
MAGDataframe = []
MAGDataframe.append(Dataframe(0, 'Time', 'min', dt=60.0, type='str',   color='black', kind='Time', format='%Y-%m-%dT%H:%M:%S'))
MAGDataframe.append(Dataframe(1, 'Bx',   'nT',  dt=60.0, type='float', color=clist_field[0], kind='Field'))
MAGDataframe.append(Dataframe(2, 'By',   'nT',  dt=60.0, type='float', color=clist_field[1], kind='Field'))
MAGDataframe.append(Dataframe(3, 'Bz',   'nT',  dt=60.0, type='float', color=clist_field[2], kind='Field'))
MAGDataframe.append(Dataframe(4, 'Btot', 'nT',  dt=60.0, type='float', color=clist_field[3], kind='Field'))
MAGDataframe.append(Dataframe(5, 'x',    'R_S', dt=60.0, type='float', color='black', kind='Coord'))
MAGDataframe.append(Dataframe(6, 'y',    'R_S', dt=60.0, type='float', color='black', kind='Coord'))
MAGDataframe.append(Dataframe(7, 'z',    'R_S', dt=60.0, type='float', color='black', kind='Coord'))
MAGDataFields['KSM'] = MAGDataframe
MAGDataFields['KSO'] = MAGDataframe

# Simulated Orbit Trajectories in KSM
SIMDataframe = []
SIMDataframe.append(Dataframe(0, 'Time', 'min', dt=60.0, type='str', color='black', kind='Time', format='%Y-%m-%dT%H:%M:%S'))
SIMDataframe.append(Dataframe(1, 'Bx', 'nT', dt=60.0, type='float', color='blue', kind='Field'))
SIMDataframe.append(Dataframe(2, 'By', 'nT', dt=60.0, type='float', color='green', kind='Field'))
SIMDataframe.append(Dataframe(3, 'Bz', 'nT', dt=60.0, type='float', color='red', kind='Field'))
SIMDataframe.append(Dataframe(4, 'Btot', 'nT', dt=60.0, type='float', color='black', kind='Field'))
SIMDataframe.append(Dataframe(5, 'x', 'R_S', dt=60.0, type='float', color='black', kind='Coord'))
SIMDataframe.append(Dataframe(6, 'y', 'R_S', dt=60.0, type='float', color='black', kind='Coord'))
SIMDataframe.append(Dataframe(7, 'z', 'R_S', dt=60.0, type='float', color='black', kind='Coord'))
SIMDataFields['KSM'] = SIMDataframe

DataFields['MAG'] = MAGDataFields
DataFields['SIM'] = SIMDataFields

# print(DataFields['MAG']['KRTP'][6].name)

SATURN_AXISROT_X = np.deg2rad(-26.7)
SATURN_AXISROT_Y = np.deg2rad(12) #Speculative
# SATURN_AXISROT_X = 0 * np.deg2rad
# SATURN_AXISROT_Y = 0 * np.deg2rad
