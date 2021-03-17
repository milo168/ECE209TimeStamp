import os
import numbers
import pandas as pd
from datetime import datetime, timedelta
import pytz

from settings import settings


# ######################################################################################################################
#
# READ EPISODE DATA FROM DATE FOLDER HOUR FILE,
# eg. /Volumes/Shibo/SHIBO/MD2K/RESAMPLE200/wild/205/CHEST/ACCELEROMETER/08-20-17/08-20-17_00.csv
#
# ######################################################################################################################

def unixtime_to_datetime(unixtime):
    """
    Convert unixtime to datetime
    Args:
        unixtime: int, unixtime

    Returns:
        datetime
    """
    if len(str(abs(unixtime))) == 13:
        return datetime.utcfromtimestamp(unixtime / 1000). \
            replace(tzinfo=pytz.utc).astimezone(settings["TIMEZONE"])
    elif len(str(abs(unixtime))) == 10:
        return datetime.utcfromtimestamp(unixtime). \
            replace(tzinfo=pytz.utc).astimezone(settings["TIMEZONE"])


def datetime_to_foldername(dt):
    return dt.strftime('%m-%d-%y')


def datetime_to_filename(dt):
    return dt.strftime('%m-%d-%y_%H.csv')


def list_date_folders_hour_files(start, end):
    """
    According to given start and end, list date folders and hour files.

    Args:
        start: int or datetime,
        end: int or datetime,

    Returns:
        list, date folders and hour files.

    """
    if isinstance(start, numbers.Integral):
        start = unixtime_to_datetime(start)
    if isinstance(end, numbers.Integral):
        end = unixtime_to_datetime(end)

    # FFList means dateFolderHourFileList
    FFList = [[datetime_to_foldername(start), datetime_to_filename(start)]]
    curr = start + timedelta(hours=1)
    while curr.replace(minute=0, second=0, microsecond=0) <= end.replace(minute=0, second=0, microsecond=0):
        FFList.append([datetime_to_foldername(curr), datetime_to_filename(curr)])
        curr += timedelta(hours=1)

    return FFList


def read_data_datefolder_hourfile(resample_path, subject, device, sensor, start_time, end_time):
    """
    Read data from date folder and hour file

    Args:
        resample_path: str, resample path
        subject: str,
        device: str,
        sensor: str,
        start_time: int,
        end_time: int,

    Returns:
        dataframe, data

    """
    # 1. read in all the data within the range
    dfConcat = []
    FFList = list_date_folders_hour_files(start_time, end_time)
    dfConcat = [pd.read_csv(os.path.join(resample_path, subject, device, sensor, FFList[i][0], FFList[i][1])) \
                for i in range(len(FFList))]
    df = pd.concat(dfConcat)

    if 'time' in df.columns.values:
        timecol = 'time'
    elif 'Time' in df.columns.values:
        timecol = 'Time'
    else:
        print('Error: time column name is neither "Time" nor "time".')
        exit()

    # 2. the starts and ends of continuous chunks in returned data
    if len(str(abs(df[timecol].iloc[0]))) == 13:
        if len(str(abs(start_time))) == 10:
            start_time = start_time * 1000
        if len(str(abs(end_time))) == 10:
            end_time = end_time * 1000
    elif len(str(abs(df[timecol].iloc[0]))) == 10:
        if len(str(abs(start_time))) == 13:
            start_time = int(start_time / 1000)
        if len(str(abs(end_time))) == 13:
            end_time = int(end_time / 1000)
    else:
        print('Error: df time column is neither 10-digit nor 13-digit unixtimestamp.')
        exit()

    df = df[(df[timecol] >= start_time) & (df[timecol] < end_time)]
    return df.reset_index(drop=True)
