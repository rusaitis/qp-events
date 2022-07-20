import datetime

# FUNCTION TO CONVERT DATETIME TO CTIME
def UnixTime(dt):
    ''' Convert datetime to UTC timestamp '''
    epoch = datetime.datetime.utcfromtimestamp(0)
    # epoch = datetime.datetime.fromtimestamp(0)
    # b = datetime.datetime(1970, 1, 1)
    delta = dt - epoch
    return delta.total_seconds()

# FUNCTION TO CONVERT CTIME TO DATETIME
def UnixTimetoDatetime(delta):
    ''' Convert UTC timestamp to datetime'''
    return datetime.datetime.utcfromtimestamp(delta)

# FUNCTION TO CONVERT DATETIME TO CTIME
def str2date(dateString, dateFormat = '%Y-%m-%dT%H:%M:%S'):
    ''' Convert a string to a datetime object '''
    return datetime.datetime.strptime(dateString, dateFormat)

def date2timestamp(date):
    ''' Convert a datetime to a timestamp '''
    return (date - datetime.datetime(1970,1,1)) / datetime.timedelta(seconds=1)
    # assert date.tzinfo is not None and date.utcoffset() is not None
    # return datetime.datetime.timestamp(date)
    # return date.strftime("%s")
    # return (date - datetime.datetime(1970,1,1, tzinfo=timezone.utc)) / timedelta(seconds=1)
    # DAY = 24*60*60 # POSIX day in seconds (exact value)
    # timestamp = (utc_date.toordinal() - date(1970, 1, 1).toordinal()) * DAY
    # timestamp = (utc_date - date(1970, 1, 1)).days * DAY

#Return the local date corresponding to the POSIX timestamp
# date.fromtimestamp(timestamp)
# datetime.fromtimestamp(timestamp, timezone.utc)
# datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=timestamp)
# https://stackoverflow.com/questions/8777753/converting-datetime-date-to-utc-timestamp-in-python
# date = str2date('2017-01-01T00:00:00')

# FUNCTION TO CONVERT DATETIME TO CTIME
# print(datetime.datetime.timestamp(date))
# date_etime = UnixTime(date.total_seconds())
# print(date_etime)
# print(date.fromtimestamp(1483257600.0))
# print(date.fromtimestamp(1483228800.0))
# processMovie2(res='5136x2880', fps=30, output='movie.mp4')
# exit()

# SIM.config["ETIME"] = 1483228800  # (01 Jan 2017)
# SIM.config["ETIME"] = 1483228900  # (01 Jan 2017)

# Unix epoch is 00:00:00 UTC on 1 January 1970
# epoch time itself (1970-01-01
# d = datetime.date(1970,1,1)
# dtt = d.timetuple() # time.struct_time(tm_year=1970, tm_mon=1, tm_mday=1, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=3, tm_yday=1, tm_isdst=-1)
# ts = time.mktime(dtt) # 28800.0
if __name__ == "__main__":
    # date = '2017-08-15T00:00:00'
    date = '2012-01-02T12:00:00'
    # Splash: 1451735970
    # Datetime1: 1325505600
    # Datetime2: 1325534400.0

    # date = '2015-01-01T08:00:00'
    # Splash: 15..
    # Datetime1: 1420099200.0
    # Datetime2: 1420128000.0.0
    date = '2012-01-08T15:34:10'
    #1: 1326036850.0 GMT
    #2: 1326065650.0 Local Time
    #3: 1326065650

    date = str2date(date)
    # print(date)
    # startTime.strftime('%Y-%m-%dT%H-%M-%S')
    ETIME = date2timestamp(date)
    print(ETIME)
