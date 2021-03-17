"""
Functionality:
    Resample time series data with linear interplation method based on 
        a). a fixed sampling rate, or
        b). a given list of sampling positions

    When resampling based on a given list of sampling positions, 
        you can merge two sensors' time series data together at common sampling positions
        
Steps:

    1. Anchor sensor: read in all continuous data block in day level,
      A. Put data files into groups according to continuity
      B. Read in one data block at a time according to continuity groups

    2. Anchor sensor: resampling within each data block
      A. Take the first entry time of a block as start time
      B. When there is a gap in the data, if gap > 0.5 s, set value as nan, otherwise take interpolation value

    3. Boat sensor: resampling anchored to anchor sensor
      A. Take the anchor sensor's time column as target time column
      B. Same as Step 2(B), when gap >0.5s, set value as nan


Action items:
1. change all namings - done
2. 'linear interplation' in description - done
3. settings -> timeColHeader - done
4. move 'if n<2: return' out of loop - done
 
"""

import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import inspect

# todo: look into np.inf to replace 0

def resample(dataDf, timeColHeader, samplingRate, gapTolerance=np.inf, fixedTimeColumn=None):
    """
    Parameters
    ----------
    dataDf : data dataframe, contains unixtime column and data column(s)

    timeColHeader : string, time column header

    samplingRate : int
        Number of samples per second

    gapTolerance: int(ms), by default np.inf
        if the distance between target point and either of the neighbors is further than gapTolerance in millisecond,
        then interpolation is nan
        if gapTolerance=0, the gapTolerance rule will not exist

    fixedTimeColumn: np.array or list, a given array(list) of sampling positions on which the resampling is based, by default None        

    Examples
    --------
    >>> timeColHeader = 'unixtime'
    >>> df = pd.DataFrame(np.arange(20).reshape(5,4),
                      columns=['unixtime', 'A', 'B', 'C'])

    >>> unix = np.array([1500000000000,1500000000048,1500000000075,1500000000100,1500000000150])
    >>> df['unixtime'] = unix
    >>> print(df)
            unixtime   A   B   C
    0  1500000000000   1   2   3
    1  1500000000048   5   6   7
    2  1500000000075   9  10  11
    3  1500000000100  13  14  15
    4  1500000000150  17  18  19
    >>> newSamplingRate = 20
    >>> newDf = resample(df, timeColHeader, newSamplingRate)
    >>> print(newDf)
            unixtime          A          B          C
    0  1500000000000   1.000000   2.000000   3.000000
    1  1500000000050   5.296295   6.296295   7.296295
    2  1500000000100  13.000000  14.000000  15.000000
    3  1500000000150  17.000000  18.000000  19.000000

    >>> newSamplingRate = 33
    >>> newDf = resample(df, timeColHeader, newSamplingRate)
    >>> print(newDf)
            unixtime          A          B          C
    0  1500000000000   1.000000   2.000000   3.000000
    1  1500000000030   3.525238   4.525238   5.525238
    2  1500000000060   6.867554   7.867554   8.867554
    3  1500000000090  11.545441  12.545441  13.545441
    4  1500000000121  14.696960  15.696960  16.696960

    (Note: the 5th unixtime is 1500000000121 instead of 1500000000120, since 5th sampling is 121.21ms away from 1st sampling.
    
    development log:
    1.
    # always take the first timestamp time[0]
    # if starttime == None:
    newSignalList = [signalArr[0]]
    newUnixtimeList = [unixtimeArr[0]]
    # else:
    #     newUnixtimeList = [starttime]
        # if starttime >= signalArr[0]
        # newSignalList = interpolate(unixtimeArr[tIndAfter-1], signalArr[tIndAfter-1], unixtimeArr[tIndAfter], signalArr[tIndAfter], t)
    
    2.
    # if gapTolerance == 0 or \
    #     ((abs(unixtimeArr[tIndAfter-1]-t)<=gapTolerance) and \
    #     (abs(unixtimeArr[tIndAfter]-t)<=gapTolerance)):

    if gapTolerance == 0 or \
        (abs(unixtimeArr[tIndAfter-1]-unixtimeArr[tIndAfter])<=gapTolerance):

    -----
    """

    originalNameOrder = list(dataDf.columns.values)
    unixtimeArr = dataDf[timeColHeader].values
    
    dataDf = dataDf.drop(timeColHeader, axis=1)
    dataArr = dataDf.values
    names = list(dataDf.columns.values)

    n = len(unixtimeArr)
    newDataList = []
    
    if n<2:
        return

    if fixedTimeColumn is None:
        deltaT = 1000.0/samplingRate
        #Looping through columns to apply the resampling method for each column
        for c in range(dataArr.shape[1]):
            signalArr = dataArr[:,c]

            # always take the first timestamp time[0]
            newSignalList = [signalArr[0]]
            newUnixtimeList = [unixtimeArr[0]]

            t = unixtimeArr[0] + deltaT
            tIndAfter = 1

            # iterate through the original signal
            while True:
                # look for the interval that contains 't'
                i0 = tIndAfter
                for i in range(i0,n):
                    if  t <= unixtimeArr[i]:#we found the needed time index
                        tIndAfter = i
                        break

                # interpolate in the right interval, gapTolenance=0 means inf tol,
                if gapTolerance == 0 or \
                    (abs(unixtimeArr[tIndAfter-1]-unixtimeArr[tIndAfter])<=gapTolerance):
                    s = interpolate(unixtimeArr[tIndAfter-1], signalArr[tIndAfter-1], \
                                    unixtimeArr[tIndAfter], signalArr[tIndAfter], t)
                else:
                    s = np.nan

                # apppend the new interpolated sample to the new signal and update the new time vector
                newSignalList.append(s)
                newUnixtimeList.append(int(t))
                # take step further on time
                t = t + deltaT
                # check the stop condition
                if t > unixtimeArr[-1]:
                    break

            newDataList.append(newSignalList)
            newDataArr = np.transpose(np.array(newDataList))

        dataDf = pd.DataFrame(data = newDataArr, columns = names)
        dataDf[timeColHeader] = np.array(newUnixtimeList)

        # change to the original column order
        dataDf = dataDf[originalNameOrder]

    else:  #if fixedTimeColumn not None:
        #Looping through columns to apply the resampling method for each column
        for c in range(dataArr.shape[1]):
            signalArr = dataArr[:,c]
            newSignalList = []
            newUnixtimeList = []

            iFixedTime = 0

            t = fixedTimeColumn[iFixedTime]
            tIndAfter = 0
            outOfRange = 1

            # iterate through the original signal
            while True:
                # look for the interval that contains 't'
                i0 = tIndAfter
                for i in range(i0,n):
                    if  t <= unixtimeArr[i]:#we found the needed time index
                        tIndAfter = i
                        outOfRange = 0
                        break

                if outOfRange:
                    s = np.nan
                else:
                    # interpolate in the right interval
                    if tIndAfter == 0: # means unixtimeArr[0] > t, there is no element smaller than t
                        s = np.nan
                    elif gapTolerance == 0 or \
                        (abs(unixtimeArr[tIndAfter-1] - unixtimeArr[tIndAfter]) <= gapTolerance):
                        s = interpolate(unixtimeArr[tIndAfter-1], signalArr[tIndAfter-1], \
                                        unixtimeArr[tIndAfter], signalArr[tIndAfter], t)
                    else:
                        s = np.nan

                # apppend the new interpolated sample to the new signal and update the new time vector
                newSignalList.append(s)
                newUnixtimeList.append(t)

                # check the stop condition
                if t > unixtimeArr[-1]:
                    break
                # take step further on time
                iFixedTime += 1

                if iFixedTime >= len(fixedTimeColumn):
                    break
                t = fixedTimeColumn[iFixedTime]

            newDataList.append(newSignalList)
            newDataArr = np.transpose(np.array(newDataList))

        dataDf = pd.DataFrame(data = newDataArr, columns = names)
        dataDf[timeColHeader] = np.array(newUnixtimeList)

        # change to the original column order
        dataDf = dataDf[originalNameOrder]
    return dataDf


def interpolate(t1, s1, t2, s2, t):
    """Interpolates at parameter 't' between points (t1,s1) and (t2,s2)
    """

    if(t1<=t and t<=t2): #we check if 't' is out of bounds (between t1 and t2)
        m = float(s2 - s1)/(t2 - t1)
        b = s1 - m*t1
        return m*t + b
    else:
        return np.nan


def list_files_in_directory(mypath):
    return [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]


def create_folder(f, deleteExisting=False):
    """
    Create a folder with option of deleting the already existing one.

    Parameters:
            f: folder path. Could be nested path (so nested folders will be created)

            deleteExising: if True then the existing folder will be deleted.

    """
    if os.path.exists(f):
        if deleteExisting:
            shutil.rmtree(f)
    else:
        os.makedirs(f)


def resample_test_case1():
    timeColHeader = 'unixtime'
    df = pd.DataFrame(np.arange(15).reshape(5,3),
                      columns=['A', 'B', 'C'])
    unix = np.array([1500000000000,1500000000048,1500000000075,1500000000349,1500000000375])
    df['unixtime'] = unix
    newDf = resample(df, timeColHeader, 20, gapTolerance=50)
    newDf = newDf.dropna(axis=0, how='any')
    print(newDf)

    fixedTimeCol = newDf['unixtime'].values

    df1 = pd.DataFrame(np.arange(100,115).reshape(5,3),
                      columns=['D', 'E', 'F'])
    unix = np.array([1499999999999,1500000000047,1500000000077,1500000000300,1500000000375])
    df1['unixtime'] = unix
    print(df1)

    newDf1 = resample(df1, timeColHeader, 20, gapTolerance=50, fixedTimeColumn=fixedTimeCol)
    newDf = newDf.set_index("unixtime")
    newDf1 = newDf1.set_index("unixtime")
    newDfConcat = pd.concat([newDf,newDf1],axis=1)
    newDfConcat = newDfConcat.dropna(axis=0, how='any')

    print(newDfConcat)


def resample_test_case2():
    timeColHeader = 'Time'

    SENSORS = ['GYROSCOPE', 'ACCELEROMETER']

    SUBJECT = 'P121'

    newSamplingRate = 20

    for SENSOR in SENSORS:
        PATH = os.path.join('CLEAN',SUBJECT,'WRIST',SENSOR,'DATA')
        OUT_PATH = os.path.join('RESAMPLE',SUBJECT,'WRIST',SENSOR,'DATA')
        create_folder(OUT_PATH)

        files = list_files_in_directory(PATH)

        for file in files:

            if not file.startswith('.'):
                df = pd.read_csv(os.path.join(PATH, file))

                newDf = resample(df, timeColHeader, newSamplingRate)
                newDf.to_csv(os.path.join(OUT_PATH, file), index=None)


def merge_test_case1():
    # ==================================================================================
    # dataframes preparation
    # ==================================================================================

    timeColHeader = 'unixtime'
    df1 = pd.DataFrame(np.arange(15).reshape(5,3),
                      columns=['A', 'B', 'C'])

    unix = np.array([1500000000000,1500000000048,1500000000075,1500000000349,1500000000375])
    df1['unixtime'] = unix

    df2 = pd.DataFrame(np.arange(100,115).reshape(5,3),
                      columns=['D', 'E', 'F'])
    unix = np.array([1499999999999,1500000000047,1500000000077,1500000000300,1500000000375])
    df2['unixtime'] = unix


    # ==================================================================================
    # df1 resampling based on a given sampling rate
    # ==================================================================================

    print('df1 before resampling:')
    print(df1)

    newDf1 = resample(df1, timeColHeader, 20, gapTolerance=50)
    newDf1 = newDf1.dropna(axis=0, how='any')

    print('\ndf1 after resampling and dropna:')
    print(newDf1)


    # ==================================================================================
    # df2 resampling based on a given list of sampling positions
    # ==================================================================================

    print('\ndf2 before resampling:')
    print(df2)

    fixedTimeCol = newDf1['unixtime'].values
    newDf2 = resample(df2, timeColHeader, 20, gapTolerance=50, fixedTimeColumn=fixedTimeCol)
    newDf2 = newDf2.dropna(axis=0, how='any')

    print('\ndf2 after resampling and dropna:')
    print(newDf2)

    # ==================================================================================
    # df1 and df2 merging
    # ==================================================================================

    newDf1 = newDf1.set_index("unixtime")
    newDf2 = newDf2.set_index("unixtime")
    newDfConcat = pd.concat([newDf1,newDf2],axis=1)
    newDfConcat = newDfConcat.dropna(axis=0, how='any')

    print('\nAfter merging and dropna:')
    print(newDfConcat)



def merge_test_case2():
    # ==================================================================================
    # dataframes preparation
    # ==================================================================================

    timeColHeader = 'unixtime'
    df1 = pd.DataFrame(np.arange(15).reshape(5,3),
                      columns=['A', 'B', 'C'])

    unix = np.array([1500000000000,1500000000048,1500000000075,1500000000349,1500000000375])
    df1['unixtime'] = unix

    df2 = pd.DataFrame(np.arange(100,115).reshape(5,3),
                      columns=['D', 'E', 'F'])
    unix = np.array([1500000000001,1500000000047,1500000000077,1500000000300,1500000000375])
    df2['unixtime'] = unix


    # ==================================================================================
    # df1 resampling based on a given sampling rate
    # ==================================================================================

    print('df1 before resampling:')
    print(df1)

    newDf1 = resample(df1, timeColHeader, 20, gapTolerance=50)
    newDf1 = newDf1.dropna(axis=0, how='any')

    print('\ndf1 after resampling and dropna:')
    print(newDf1)


    # ==================================================================================
    # df2 resampling based on a given list of sampling positions
    # ==================================================================================

    print('\ndf2 before resampling:')
    print(df2)

    fixedTimeCol = newDf1['unixtime'].values
    newDf2 = resample(df2, timeColHeader, 20, gapTolerance=50, fixedTimeColumn=fixedTimeCol)
    newDf2 = newDf2.dropna(axis=0, how='any')

    print('\ndf2 after resampling and dropna:')
    print(newDf2)

    # ==================================================================================
    # df1 and df2 merging
    # ==================================================================================

    newDf1 = newDf1.set_index("unixtime")
    newDf2 = newDf2.set_index("unixtime")
    newDfConcat = pd.concat([newDf1,newDf2],axis=1)
    newDfConcat = newDfConcat.dropna(axis=0, how='any')

    print('After merging and dropna:')
    print(newDfConcat)


def resample_folder(inpath, outpath, timeColHeader, newSamplingRate, gapTolerance=np.inf, fixedTimeColumn=None):
    create_folder(outpath)
    files = list_files_in_directory(inpath)

    for file in files:
        if not file.startswith('.'):
            df = pd.read_csv(os.path.join(inpath, file))
            print(df)
            df = df.drop(columns=['date'])
            print(df.dtypes)
            # df = df.astype({"Time": float})
            newDf = resample(df, timeColHeader, newSamplingRate,\
                            gapTolerance=gapTolerance, fixedTimeColumn=fixedTimeColumn)
            newDf.to_csv(os.path.join(outpath, file), index=None)



if __name__ == '__main__':
    # resample_test_case1()
    # resample_test_case2()
    merge_test_case1()
    # merge_test_case2()


