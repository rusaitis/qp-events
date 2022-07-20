import datetime
import numpy as np
import math
import os

class Crossing():
    def __init__(self,
                 datetime=None,
                 Type=None,
                 Direction=None,
                 x=None,
                 y=None,
                 z=None,
                 ):
        """ Boundary Crossing class """
        self.datetime = datetime
        self.Type = Type
        self.Direction = Direction
        self.x = x
        self.y = y
        self.z = z

def export_boundary_crossings(
        filepath = '/Users/leo/DATA/cassini-crossings/BS_MP_Crossing_List.txt',
        output = '/Users/leo/DATA/cassini-crossings/CROSSINGS.npy',
        interval_in_sec = 60*60): 

    crossings = []
    with open(filepath) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            nl = line.split()
            year = int(nl[0])
            doy = int(nl[1])
            h = int(nl[2])
            mins = int(nl[3])
            Typ = nl[4]
            Direction = nl[5]
            x = float(nl[6])
            y = float(nl[7])
            z = float(nl[8])
            dt = datetime.datetime.strptime('%d %03d %02d:%02d'
                                            %(year, doy, h, mins),
                                            '%Y %j %H:%M')
            crossings.append(Crossing(datetime=dt,
                                      Type=Typ,
                                      Direction=Direction,
                                      x=x, y=y, z=z))
            line = fp.readline()
            cnt += 1
    print('Number of Boundary Crossing Events: ', len(crossings))

    T0 = '2004-06-30T00:00:00'
    T1 = '2017-09-15T00:00:00'
    dateMissionFrom = datetime.datetime.strptime(T0, '%Y-%m-%dT%H:%M:%S')
    dateMissionTo = datetime.datetime.strptime(T1, '%Y-%m-%dT%H:%M:%S')

    TIMES = []
    time = dateMissionFrom
    while True:
        time += datetime.timedelta(seconds=interval_in_sec)
        TIMES.append(time)
        if time >= dateMissionTo:
            break
    TIMES = np.asarray(TIMES)
    LOCS = np.full(len(TIMES), 9)
    CROS = np.full(len(TIMES), None)
    CROSSINGS = np.vstack((TIMES, LOCS, CROS))

    ind_prev = 0
    for c in crossings:
        if c.Type + c.Direction == 'BSI':
            loc = 1
        elif c.Type + c.Direction == 'BSO':
            loc = 2
        elif c.Type + c.Direction == 'MPI':
            loc = 0
        elif c.Type + c.Direction == 'MPO':
            loc = 1
        elif c.Direction == 'E_SW':
            loc = 2
        elif c.Direction == 'E_SH':
            loc = 1
        elif c.Direction == 'E_SP':
            loc = 0
        elif c.Direction == 'E_MP':
            loc = 0
        elif c.Direction[0] == 'S':
            loc = 9
        for i in range(ind_prev, len(CROSSINGS[0])):
            if CROSSINGS[0][i] >= c.datetime:
                CROSSINGS[1][i:-1] = loc
                if c.Type in ['BS', 'MP']:
                    CROSSINGS[2][i] = c.Type + c.Direction
                ind_prev = i
                break

    np.save(output, CROSSINGS)
