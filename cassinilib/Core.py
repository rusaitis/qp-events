import numpy as np

RAD = np.pi / 180.
DEG = 180. / np.pi

def NearestOdd(f):
    ceil = np.ceil(f) // 2 * 2 + 1
    floor = np.floor(f) // 2 * 2 + 1
    if abs(f-ceil) <= 0.5:
        return ceil
    else:
        return floor

def AvgArray(arr, dt):
    dt = int(dt)
    arr = np.asarray(arr)
    arr = arr.astype(float)
    padright = (dt - arr.size%dt) % dt
    # print('arr.size mod dt:', arr.size%dt)
    # print('padright:', padright)
    padded = np.pad(arr, (0, padright), mode='constant', constant_values=np.NaN)
    # print('padded:', padded)
    return np.nanmean(padded.reshape(-1, dt), axis=1)

if __name__ == "__main__":
    # a = [1, 2, 3, 4, np.NaN]
    aa = [1, 2, 3, np.NaN, np.NaN, 4]
    # a = np.asarray(a)
    # b = AvgArray(a, 10)
    # print(b)
    print(aa)
    print(AvgArray(aa, 11))
    # print(np.nanmean(aa))
