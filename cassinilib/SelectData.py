import datetime
from cassinilib.FileCheck import FileCheck
import datetime

def findReadResolution(data_path,
                       datetimeFormat,
                       resolutionInSec = 1.,
                       dateColumn = 0):
    ''' Find read sampling resolution for the data '''
    if FileCheck(data_path):
        with open(data_path) as file:
            time0 = file.readline().split()[dateColumn]
            time1 = file.readline().split()[dateColumn]
            time0 = datetime.datetime.strptime(time0, datetimeFormat)
            time1 = datetime.datetime.strptime(time1, datetimeFormat)
            timedelta = (time1 - time0).total_seconds()
    return int(resolutionInSec // timedelta)

def readTimeseriesFile(data_path,
                       datetimeFormat,
                       dateFrom = None,
                       dateTo = None,
                       dateColumn = 0,
                       resolutionInSec = None):
    ''' Read data for a given data file '''
    if FileCheck(data_path):

        if resolutionInSec:
            readRes = findReadResolution(data_path,
                                         datetimeFormat,
                                         resolutionInSec=resolutionInSec,
                                         dateColumn = dateColumn)
        else:
            readRes = 1

        with open(data_path) as file:
            dataSelected = []
            for n, line in enumerate(file):
                if n % readRes == 0:
                    data = line.split()
                    date = datetime.datetime.strptime(data[dateColumn], datetimeFormat)
                    if dateFrom and dateTo:
                        if date >= dateFrom and date <= dateTo:
                            dataSelected.append(data)
                        elif date > dateTo:
                            break
                    else:
                        dataSelected.append(data)
    return dataSelected

def checkDatetimeType(variable,
                      datetimeFormat):
    ''' Check if a variable is already a datetime object '''
    if type(variable) is datetime.datetime:
        return variable
    else:
        return datetime.datetime.strptime(variable, datetimeFormat)

def generateFilePaths(basestring,
                      dateFrom,
                      dateTo):
    ''' Generate a Cartesian Product of all possible paths
    based on basestring '''
    paths = []
    yearFrom = int(dateFrom.strftime('%Y'))
    yearTo = int(dateTo.strftime('%Y'))
    dayFrom = dateFrom.timetuple().tm_yday
    dayTo = dateTo.timetuple().tm_yday
    for year in range(yearFrom, yearTo + 1):
        path = basestring.replace('YYYY', str(year))
        if 'DDD' in basestring:
            for day in range(dayFrom if year == yearFrom else 1,
                             dayTo if year == yearTo else dayEnd + 1):
                path = basestring.replace('DDD', "%03d" % (day))
                paths.append(path)
        else:
            paths.append(path)
    return paths

def SelectData(dataFilePath,
               dateFrom,
               dateTo,
               datetimeFormat = '%Y-%m-%dT%H:%M:%S',
               marginInSec = 0,
               resolutionInSec = None):
    ''' Select Data for a given datetime interval '''
    margin = datetime.timedelta(seconds = marginInSec)
    dateFrom = checkDatetimeType(dateFrom, datetimeFormat) - margin
    dateTo = checkDatetimeType(dateTo, datetimeFormat) + margin
    dataSelected = []

    paths = generateFilePaths(dataFilePath, dateFrom, dateTo)

    for path in paths:
        if FileCheck(path):
            newData = readTimeseriesFile(path,
                                         datetimeFormat,
                                         dateFrom = dateFrom,
                                         dateTo = dateTo,
                                         resolutionInSec = resolutionInSec)
            dataSelected += newData

    return dataSelected
#===============================================================================
