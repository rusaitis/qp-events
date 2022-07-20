import numpy as np
import datetime


class Event:
    ''' A class for all the event information '''
    def __init__(self,
        datefrom = datetime.datetime.strptime('2000-01-01T00:00:00', '%Y-%m-%dT%H:%M:%S'),
        dateto = datetime.datetime.strptime('2000-01-01T10:00:00', '%Y-%m-%dT%H:%M:%S'),
        Waves = [],
        COORD = None,
        COORD_LABEL = None,
        LT = None,
        comment = None,
        axis = None,
        period = None,
        amplitude = None,
        period_std = None,
        amplitude_std = None,
        density = None,
        MagLAT = None,
        R = None,
        L = None,
        SNR = None,
        closedFL = None,
        synthetic = False
        ):
        self.datefrom = datefrom
        self.dateto = dateto
        self.Waves = Waves
        self.axis = axis
        self.period = period
        self.amplitude = amplitude
        self.period_std = period_std
        self.amplitude_std = amplitude_std
        self.COORD = COORD
        self.COORD_LABEL = COORD_LABEL
        self.LT = LT
        self.density = density
        self.MagLAT = MagLAT
        self.R = R
        self.L = L
        self.comment = comment
        self.SNR = SNR
        self.closedFL = closedFL
        self.synthetic = synthetic

    def is_significant(self, SNR=None):
        if SNR:
            return self.SNR > SNR
        else:
            return self.SNR > 3


def printEvent(event, margin=None, n=None, N=None):
    ''' Print Current Event information '''
    if margin is not None:
        DATE_FROM = event.datefrom - datetime.timedelta(seconds=TIME_MARGIN)
        DATE_TO = event.dateto + datetime.timedelta(seconds=TIME_MARGIN)
    else:
        DATE_FROM = event.datefrom
        DATE_TO = event.dateto
    DATE_YEAR = DATE_FROM.strftime('%Y')
    EVENT_DURATION = event.dateto - event.datefrom
    if n is not None:
        print('GATHERING DATA FOR EVENT ' + str(n+1) + '/' + str(N))
    print('EVENT DATE: ' + event.datefrom.strftime('%Y-%m-%d %H:%M') +
    ' - ' + event.dateto.strftime('%Y-%m-%d %H:%M') + ' (' +
    str(round(EVENT_DURATION.total_seconds()/60./60.,2)) + ' hours)')
