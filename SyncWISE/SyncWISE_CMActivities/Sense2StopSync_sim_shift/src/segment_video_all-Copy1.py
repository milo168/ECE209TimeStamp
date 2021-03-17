import os
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from load_sensor_data import read_data_datefolder_hourfile
from settings import settings
from utils import csv_read
from scipy.interpolate import CubicSpline


FPS = settings["FPS"]
FRAME_INTERVAL = settings["FRAME_INTERVAL"]
sample_counts = settings["sample_counts"]
flow_path = settings["flow_path"]

def load_start_time(start_time_file, vid):
    """
    load start time

    Args:
        start_time_file: str
        vid: str, video

    Returns:
        int, start time

    """
    df_start_time = csv_read(start_time_file).set_index("video_name")
    if vid not in df_start_time.index:
        print("Error: ", vid, " not in ", start_time_file)
        exit()
    start_time = df_start_time.loc[vid]["start_time"]
    return int(start_time)


def reliability_df_to_consecutive_seconds(
    df_sensor_rel, window_size_sec, stride_sec, threshold=sample_counts
):
    """
    Convert from reliability df to consecutive seconds represented with start and end time.

    Args:
        df_sensor_rel: dataframe, sensor reliability
        window_size_sec:, int, window_size
        stride_sec: int, stride
        threshold: float

    Returns:
        win_start_end: a list of all the possible [window_start, window_end] pairs.
    """
    # use the threshold criterion to select 'good' seconds
    rel_seconds = (
        df_sensor_rel[df_sensor_rel["SampleCounts"] > threshold]
        .sort_values(by="Time")["Time"]
        .values
    )
    win_start_end = consecutive_seconds(rel_seconds, window_size_sec, stride_sec)
    return win_start_end


def consecutive_seconds(rel_seconds, window_size_sec, stride_sec=1):
    """
    Return a list of all the possible [window_start, window_end] pairs 
        containing consecutive seconds of length window_size_sec inside.
    Args:
        rel_seconds: a list of qualified seconds
        window_size_sec: int
        stride_sec: int
    Returns:
        win_start_end: a list of all the possible [window_start, window_end] pairs.

    Test:
        >>> rel_seconds = [2,3,4,5,6,7,9,10,11,12,16,17,18]; window_size_sec = 3; stride_sec = 1
        >>> print(consecutive_seconds(rel_seconds, window_size_sec))
        >>> [[2, 4], [3, 5], [4, 6], [5, 7], [9, 11], [10, 12], [16, 18]]
    """
    win_start_end = []
    for i in range(0, len(rel_seconds) - window_size_sec + 1, stride_sec):
        if rel_seconds[i + window_size_sec - 1] - rel_seconds[i] == window_size_sec - 1:
            win_start_end.append([rel_seconds[i], rel_seconds[i + window_size_sec - 1]])
    return win_start_end


def load_flow(vid_path, fps, start_time, offset_sec=0):
    """
    load flow data

    Args:
        vid_path: str, video path
        fps: float
        start_time: int
        offset_sec: int

    Returns:
        dataframe, flow data dataframe
        int, length of video in ms

    """
    motion = pickle.load(open(vid_path, "rb"))
    frame_len = 1000.0 / fps  # duration of a frame in ms
    frames = motion.shape[0]  # number of frames
    len_ms = frames * frame_len  # duration of all frames in ms
    timestamps_int = np.arange(
        start_time + offset_sec * 1000,
        start_time + offset_sec * 1000 + len_ms,
        frame_len,
    ).astype(int)
    l = min(len(timestamps_int), motion.shape[0])
    timestamps_int = timestamps_int[:l]
    motion = motion[:l, :]
    df_flow = pd.DataFrame(
        {"time": timestamps_int, "flowx": motion[:, 0], "flowy": motion[:, 1]}
    )
    df_flow["second"] = (df_flow["time"] / 1000).astype(int)
    df_flow["diff_flowx"] = df_flow["flowx"].diff()
    df_flow["diff_flowy"] = df_flow["flowy"].diff()
    df_flow = df_flow.reset_index()
    return df_flow, len_ms

def process_flow(flow, fps, start_time, offset_sec=0):
    """
    load flow data

    Args:
        vid_path: str, video path
        fps: float
        start_time: int
        offset_sec: int

    Returns:
        dataframe, flow data dataframe
        int, length of video in ms

    """
    motion = flow
    frame_len = 1000.0 / fps  # duration of a frame in ms
    frames = motion.shape[0]  # number of frames
    len_ms = frames * frame_len  # duration of all frames in ms
    timestamps_int = np.arange(
        start_time + offset_sec * 1000,
        start_time + offset_sec * 1000 + len_ms,
        frame_len,
    ).astype(int)
    l = min(len(timestamps_int), motion.shape[0])
    timestamps_int = timestamps_int[:l]
    motion = motion[:l, :]
    df_flow = pd.DataFrame(
        {"time": timestamps_int, "flowx": motion[:, 0], "flowy": motion[:, 1]}
    )
    df_flow["second"] = (df_flow["time"] / 1000).astype(int)
    df_flow["diff_flowx"] = df_flow["flowx"].diff()
    df_flow["diff_flowy"] = df_flow["flowy"].diff()
    df_flow = df_flow.reset_index()
    return df_flow, len_ms

def load_sensors_cubic(
    raw_path, sub, device, sensors, sensor_col_header, start_time, end_time, fps
):
    """
    load sensor data with cubic spline resampling

    Args:
        raw_path: str,
        sub: str, subject
        device: str
        sensors: list, sensors
        sensor_col_header: list of sensor column headers
        start_time: int
        end_time: int
        fps: float

    Returns:
        dataframe, sensor data

    """
    df_list = []
    for s, col in zip(sensors, sensor_col_header):
        df_sensor = read_data_datefolder_hourfile(
            raw_path, sub, device, s, start_time, end_time
        )
        df_sensor = df_sensor[["time", col]]
        df_sensor["time"] = pd.to_datetime(df_sensor["time"], unit="ms")
        df_sensor = df_sensor.set_index("time")
        df_resample = df_sensor.resample(FRAME_INTERVAL).mean()  
        # FRAME_INTERVAL as 0.03336707S is the most closest value to 1/29.969664 pandas accepts
        df_resample = df_resample.interpolate(method="spline", order=3) # cubic spline interpolation
        df_list.append(df_resample)
    df_sensors = pd.concat(df_list, axis=1)
    return df_sensors


def pca_sensor_flow(df_sensor, df_flow):
    """
    Calculate the pca of sensor data and flow data separately

    Args:
        df_sensor: dataframe, sensor data
        df_flow:dataframe, flow data

    Returns:
        dataframe, sensor data with pca column
        dataframe, flow data with pca column
    """
    # save all data, may used in pca or xx when compute drift; code/syncwise/drift_confidence.py#L8
    pca_sensor = PCA(n_components=1)
#     print(np.shape(df_sensor))
#     print(df_sensor[0:5][:])
    df_sensor[["accx", "accy", "accz"]] -= df_sensor[["accx", "accy", "accz"]].mean()
    df_sensor["acc_pca"] = pca_sensor.fit_transform(
        df_sensor[["accx", "accy", "accz"]].to_numpy()
    )
    diffflow_mat = df_flow[["diff_flowx", "diff_flowy"]].to_numpy()
    diffflow_mat -= np.mean(diffflow_mat, axis=0)
    pca_diffflow = PCA(n_components=1)
    df_flow["diffflow_pca"] = pca_diffflow.fit_transform(diffflow_mat)
#     print(df_sensor[0:5][:])
#     print(df_flow[0:5][:])
#     print(df_flow[0:2][1:3])
#     print(df_flow["diffflow_pca"][1:3])
#     print(df_flow[["diff_flowx", "diff_flowy"]][1:3])

    return df_sensor, df_flow


def add_win_rand_offset(
    df_sensor,
    df_flow,
    vid_name,
    win_start_end,
    start_time,
    end_time,
    kde_num_offset,
    kde_max_offset,
    window_size_sec,
    window_criterion,
    fps
):
    """
    add random offset to each window

    Args:
        df_sensor: dataframe, sensor data
        df_flow: dataframe, flow data
        vid_name: str, video name
        win_start_end: list
        start_time: int
        end_time: int
        kde_num_offset: int
        kde_max_offset: int
        window_size_sec: int
        window_criterion: float
        fps: float

    Returns:
        int, count of windows
        list, a list of all dataframes of videos
        list, a list of all video data information

    """
    df_dataset_vid = []
    info_dataset_vid = []
    cnt_windows = 0
    # add an offset to each window sensor-video pair
    for pair in win_start_end:
#         start = pair[0] * 1000
#         end = pair[1] * 1000 + 1000
        start = pair[0] # ms
        end = pair[1] + 1 # ms
        df_window_sensor = df_sensor[
            (df_sensor["time"] >= pd.to_datetime(start, unit="ms"))
            & (df_sensor["time"] < pd.to_datetime(end, unit="ms"))
        ]
        for i in range(kde_num_offset):
            # match video dataframe
            offset = random.randint(-kde_max_offset, kde_max_offset)
            offset = (  # in the range of the origin data
                min(offset, end_time - end)
                if offset > 0
                else max(offset, start_time - start)
            )
            df_window_flow = df_flow[
                (df_flow["time"] >= pd.to_datetime(start + offset, unit="ms"))
                & (df_flow["time"] < pd.to_datetime(end + offset, unit="ms"))
            ]
            pd.options.mode.chained_assignment = None
            df_window_flow.loc[:, "time"] = df_window_flow.loc[:, "time"] \
                - pd.Timedelta(offset, unit="ms") # change the time index of the shifted data
            df_window = pd.merge_asof(
                df_window_sensor,
                df_window_flow,
                on="time",
                tolerance=pd.Timedelta("29.969664ms"),
                direction="nearest",
            ).set_index("time")
            df_window = df_window.dropna(how="any") #‘any’ : If any NA values are present, drop that row or column.
            if len(df_window) > fps * window_size_sec * window_criterion:

                cnt_windows += 1
                df_dataset_vid.append(df_window)
                info_dataset_vid.append(
                    [vid_name, start, end, offset]
                )  # relatively video name, sensor starttime, sensor endtime, video offset
    return cnt_windows, df_dataset_vid, info_dataset_vid


def seg_smk_video(
#     subject,
    video,
    flow,
    sensor,
    window_size_sec,
    stride_sec,
    offset_sec,
    kde_num_offset,
    kde_max_offset,
    window_criterion,
#     starttime_file,
    fps,
):
    """
    Segment one smoking video.

    Args:
        subject: str
        video: str
        window_size_sec: int
        stride_sec: int
        offset_sec: int
        kde_num_offset: int
        kde_max_offset: int
        window_criterion: float
        starttime_file: str
        fps: float

    Returns:
        list, a list of (video name, count of windows) pairs
        list, a list of all dataframes of videos
        list, a list of all video data information

    """
    # ==========================================================================================
    reliability_resample_path = settings['reliability_resample_path']
    raw_path = settings['raw_path']
    flow_path = settings['flow_path']
    # ==========================================================================================

    vid_name = video
    vid_qual_win_cnt = []
    df_dataset = []
    info_dataset = []

#     device = "CHEST"
#     sensor = "ACCELEROMETER"
#     sensors = ["ACCELEROMETER_X", "ACCELEROMETER_Y", "ACCELEROMETER_Z"]
#     sensor_col_header = ["accx", "accy", "accz"]
#     flow_path = os.path.join(
#         flow_path, "sub{}".format(subject), vid_name + ".pkl"
#     )

    # load start end time
#     start_time = load_start_time(starttime_file, vid_name)
    start_time = 0

    # load optical flow data and assign unixtime to each frame
#     df_flow, len_ms = load_flow(flow_path, fps, start_time, offset_sec)
#  need modify
#     flow = pickle.load(open(flow_path, "rb"))
    df_flow, len_ms = process_flow(flow, fps, start_time, offset_sec)
    end_time = int(start_time) + int(len_ms)

#     # load sensor reliability data
#     df_sensor_rel = read_data_datefolder_hourfile(
#         reliability_resample_path,
#         subject,
#         device,
#         sensor + "_reliability",
#         start_time,
#         end_time,
#     )

#     # manually set all the reliability to 10
#     tmp_start_time = int(start_time / 1000)
#     tmp_end_time = int(end_time / 1000)
#     tmp_Time = np.array(range(tmp_start_time, tmp_end_time, 1))
#     tmp_SampleCounts = 10*np.ones(tmp_Time.shape) # set all scores to 10
    
#     tmp_df_sensor_rel = {
#        'Time':tmp_Time,
#        'SampleCounts':tmp_SampleCounts}
#     tmp_df_sensor_rel = pd.DataFrame(tmp_df_sensor_rel)
    
# #     print(type(tmp_df_sensor_rel))
# #     print(np.shape(tmp_df_sensor_rel), tmp_start_time, tmp_end_time)
# #     print(tmp_df_sensor_rel[0:10][:])
# #     print('\n', tmp_df_sensor_rel[330:][:])



#     # record consecutive seconds of the length the same as window_size
#     win_start_end = reliability_df_to_consecutive_seconds(
# #         df_sensor_rel, window_size_sec, stride_sec, threshold=7
#         tmp_df_sensor_rel, window_size_sec, stride_sec, threshold=7
#     )
#     print('origin', win_start_end)


    

    # generate win_start_end directly; change from sec to mili-sec ms
    win_start_end = []
    stride_ms = int(stride_sec*1000)
    window_size_ms = int(window_size_sec*1000)
    for i in range(0, end_time - start_time + 1  - window_size_ms + 1, stride_ms): # suppose stride_sec is ms
        win_start_end.append([i+start_time, i + window_size_ms - 1+start_time])
#     print('modify', win_start_end)
#     print('start_time', start_time)
    
    
    
    
    ## extract the optical flow frames of the good seconds according to sensor data
    df_flow["time"] = pd.to_datetime(df_flow["time"], unit="ms")
    df_flow = df_flow[["flowx", "flowy", "diff_flowx", "diff_flowy", "time"]].set_index(
        "time"
    )


    # extract the raw data 'ACCELEROMETER_X' (,'ACCELEROMETER_Y', 'ACCELEROMETER_Z') of consecutive chunk and resample
    #   according to video frame timestamp.
    # note that we need to resample sensor data according to video time,
    #   so the input of resample function here should be raw data instead of already sampled data to avoid resample twice.

#     df_sensors = load_sensors_cubic(
#         raw_path,
#         subject,
#         device,
#         sensors,
#         sensor_col_header,
#         start_time,
#         end_time,
#         fps,
#     )
#     print("df_sensors", df_sensors.shape)
#     print((df_sensors[0:3]))
#     assume we have sensor data after resample
#     assume sensor data same length as flows
#     df_sensors_notime = df_sensors.iloc[:,0:3].to_numpy()

#     # write df_sensors_notime np to .pkl
#     fileObject = open(os.path.join(raw_path, "df_sensors_notime.pkl"), 'wb')
#     pickle.dump(df_sensors_notime, fileObject)
#     fileObject.close()    

# CubicSpline the sensor data (40,3) to (44,3)
#     print(sensor.shape)
    x = np.arange(sensor.shape[0])
    x2 = np.arange(0, sensor.shape[0], sensor.shape[0]/df_flow.shape[0])
    sensor_CS = []
    #accx
    y = sensor[:,0]
    CS = CubicSpline(x, y)
    y2 = CS(x2)
#     print('sensor_CS',np.shape(sensor_CS))
#     print('y2',np.shape(y2))
    sensor_CS = np.c_[y2]
    
#     #accy
#     y = sensor[:,1]
#     CS = CubicSpline(x, y)
#     y2 = CS(x2)
# #     sensor_CS = np.c_[y2]
#     sensor_CS = np.insert(sensor_CS, 0, values=y2, axis=1)
#     #accy
#     y = sensor[:,2]
#     CS = CubicSpline(x, y)
#     y2 = CS(x2)
# #     sensor_CS = np.c_[y2]
#     sensor_CS = np.insert(sensor_CS, 0, values=y2, axis=1)
#     print('sensor_CS',np.shape(sensor_CS))
    for cs_id in range(1,sensor.shape[1]):
        y = sensor[:,cs_id]
        CS = CubicSpline(x, y)
        y2 = CS(x2)
        sensor_CS = np.insert(sensor_CS, 0, values=y2, axis=1)
    print('sensor_CS',np.shape(sensor_CS))
    


#   need modify; load sensor data from .pkl
    df_sensors_notime = sensor_CS
#     df_sensors_notime = pickle.load(open(os.path.join(raw_path, "df_sensors_notime.pkl"), 'rb'))
    print("load df_sensors_notime", df_sensors_notime.shape)    


    # concatenate df_sensors and df_flow
#     df_resample = pd.merge_asof(
#         df_flow,
#         df_sensors,
#         on="time",
#         tolerance=pd.Timedelta("30ms"),
#         direction="nearest",
#     ).set_index("time")
#     print("df_resample merge_asof", df_resample.shape)
#     print(df_resample[0:3])
#need modify; add sensor data (no time index) in np format to flow (with time index) in dataframe
    df_resample_merge = df_flow.copy()
    df_resample_merge.loc[:,'accx'] = pd.Series(df_sensors_notime[:df_flow.shape[0], 0], index=df_flow.index)
    df_resample_merge.loc[:,'accy'] = pd.Series(df_sensors_notime[:df_flow.shape[0], 1], index=df_flow.index)
    df_resample_merge.loc[:,'accz'] = pd.Series(df_sensors_notime[:df_flow.shape[0], 2], index=df_flow.index)
    print("df_resample_merge", df_resample_merge.shape)
    print(df_resample_merge[0:3])
    df_resample_merge = df_resample_merge[1:] #df_flow first row has NaN
    print("df_resample_merge", df_resample_merge.shape)
    print(df_resample_merge[0:3])
    df_resample = df_resample_merge.copy()
    
    df_resample = df_resample.dropna(how="any") # drop NaN

#     print('There are {} points left (freq=30Hz) for sensor-opticalflow combined.\n'.format(len(df_resample)))
#need modify to remove accx, accy, accz; there 12 channels;
#     df_sensor = df_resample[["accx", "accy", "accz"]].reset_index() #keep the data with such index;reset_index 0,1,2,...
#     df_sensor = df_resample.iloc[:,4:7].reset_index() #keep the data with such index
#     print("df_sensor")
#     print((df_sensor[0:3]))
#     df_sensor_index = df_resample.iloc[:,4:7].reset_index() #keep the data with such index
#     print("df_sensor_index")
#     print((df_sensor_index[0:3]))
#need modify to remove accx, accy, accz; there 12 channels;
#     df_sensor_np = df_sensor[["accx", "accy", "accz"]].to_numpy()
#     df_sensor_np = df_resample.iloc[:,4:7].to_numpy()
    
    #"diff_flowx", "diff_flowy" is okay, same format
    print("df_flow",df_flow.shape)
    print(df_flow[0:3])
#     df_flow = df_resample[["flowx", "flowy", "diff_flowx", "diff_flowy"]].reset_index()
#     df_flow = df_flow[["flowx", "flowy", "diff_flowx", "diff_flowy"]].reset_index()
    print("df_flow",df_flow.shape)
    print(df_flow[0:3])
    #"diff_flowx", "diff_flowy" is okay, same format
#     df_flow_np = df_flow[["diff_flowx", "diff_flowy"]].to_numpy()    
   #     # PCA with zero-mean the data using accx, accy, accz
#     df_sensor, df_flow = pca_sensor_flow(df_sensor, df_flow)


#     df_flow["time"] = pd.to_datetime(df_flow["time"], unit="ms")
#     df_flow = df_flow[["flowx", "flowy", "diff_flowx", "diff_flowy", "time"]].set_index("time")
    df_flow = df_flow.reset_index()
    df_flow = df_flow[1:]
#     print(df_flow[0:3])
    df_sensor_direct = df_flow["time"].to_frame()
#     print("df_sensor_direct", df_sensor_direct.shape)
#     print(df_sensor_direct[0:3])
    pca_sensor = PCA(n_components=1)
    df_sensor_np = sensor_CS[1:]
    df_sensor_direct["acc_pca"] = pca_sensor.fit_transform(df_sensor_np)
    df_sensor_direct = df_sensor_direct.set_index("time")
#     print(df_sensor_direct.shape)
#     print(df_sensor_direct[0:3])
#     df_sensor = df_sensor_direct
    
    df_sensor = df_sensor_direct.reset_index().copy()

    
    # PCA with zero-mean the data
#     pca_df_sensor, pca_df_flow = pca_sensor_flow_np(df_sensor_np, df_flow_np)
#     pca_sensor = PCA(n_components=1)
#     df_sensor_np -= df_sensor_np.mean()
#     df_sensor["acc_pca"] = pca_sensor.fit_transform(df_sensor_np)
#     print(df_sensor[0:3])  
#     df_sensor = df_sensor_direct.reset_index().copy()
#     print(df_sensor[0:3])

#     df_flow_np = df_flow[["diff_flowx", "diff_flowy"]].to_numpy()    
    diffflow_mat = df_flow[["diff_flowx", "diff_flowy"]].to_numpy()
    diffflow_mat -= np.mean(diffflow_mat, axis=0)
    pca_diffflow = PCA(n_components=1)
    df_flow["diffflow_pca"] = pca_diffflow.fit_transform(diffflow_mat)

    
    ## select anchor windows from sensor, apply shifts in videos
    cnt_windows, df_dataset_vid, info_dataset_vid = add_win_rand_offset(
        df_sensor,
        df_flow,
        vid_name,
        win_start_end,
        start_time,
        end_time,
        kde_num_offset,
        kde_max_offset,
        window_size_sec,
        window_criterion,
        fps
    )
    print("for video", vid_name, ":", cnt_windows, "/", len(win_start_end), "(*kde_num_offset",kde_num_offset,")", "windows left (after remove NaN according to window_criterion)")
    df_dataset += df_dataset_vid
    info_dataset += info_dataset_vid
    vid_qual_win_cnt.append((vid_name, cnt_windows))

    return vid_qual_win_cnt, df_dataset, info_dataset


def segment_video_all(
    window_size_sec,
    stride_sec,
    offset_sec,
    kde_num_offset,
    kde_max_offset,
    window_criterion,
    # data_dir,
#     starttime_file,
    fps=FPS,
):
    """
    Segment all videos

    Args:
        window_size_sec: int, window size
        stride_sec: int, stride
        offset_sec: float, offset
        kde_num_offset: int, number of offsets in KDE algorithm
        kde_max_offset: int, max offset in KDE algorithm
        window_criterion: float, window criterion
        starttime_file:, str, start time file
        fps: float

    Returns:
         list, a list of all dataframes of videos
         list, a list of all video data information
    """
#     flow_path = os.path.join("/home/gaofeng/Courses/209AS_AIoT/simpleSyncWISE/Sense2StopSync/flow_pwc/", "sub{}".format(234),  "234 GOPR0523.pkl")
#     flows = pickle.load(open(flow_path, "rb"))
#     print("flows", flows.shape)
    
#     sensors = pickle.load(open(os.path.join("/home/gaofeng/Courses/209AS_AIoT/simpleSyncWISE/Sense2StopSync/RAW/wild/df_sensors_notime.pkl"), 'rb'))
#     print("sensors", sensors.shape)
    
    
    # load data form TimaAwareness
    flow_path = os.path.join("/home/gaofeng/Courses/209AS_AIoT/TimeAwareness/Data_withVideo/flows_2d_test_videos.pkl")
    flows = pickle.load(open(flow_path, "rb"))
    print("flows", flows.shape)
    
    
    Test_data = pickle.load(open(os.path.join("/home/gaofeng/Courses/209AS_AIoT/TimeAwareness/Data_withVideo/Data_test_71.pkl"), 'rb'))
    sensors = Test_data[0]
    print("sensors", np.shape(sensors))
    sensors_xyz = sensors[:,:,0:3]
    print("sensors_xyz", np.shape(sensors_xyz))

    #     df_start_time = csv_read(starttime_file).set_index("video_name")
#     video_names = df_start_time.index.tolist()
#     print("video_names", np.shape(video_names))
#     print(video_names)
    video_names = []
    for vid in range(1):
        video_names.append(str(vid))
#     video_names = str(video_names)
#     print("video_names", np.shape(video_names))
#     print(video_names)
#     print(df_start_time, video_names)

#     exit()

    df_dataset_all = []
    info_dataset_all = []
    vid_qual_win_cnt_all = []

#     for video_name in video_names:
#         subject = video_name[:3]
#         video = video_name[4:]
#         vid_qual_win_cnt, df_dataset, info_dataset = seg_smk_video(
#             subject=subject,
#             video=video,
#             window_size_sec=window_size_sec,
#             stride_sec=stride_sec,
#             offset_sec=offset_sec,
#             kde_num_offset=kde_num_offset,
#             kde_max_offset=kde_max_offset,
#             window_criterion=window_criterion,
#             starttime_file=starttime_file,
#             fps=fps,
#         )

#         vid_qual_win_cnt_all += vid_qual_win_cnt
#         df_dataset_all += df_dataset
#         info_dataset_all += info_dataset
    
    for video_name in video_names:
#         subject = video_name[:3]
#         video = video_name[4:]
        video = video_name
        flow = flows[int(video_name)]
        sensor = sensors_xyz[int(video_name)]
        # CubicSpline the sensor from 40 to 44(len of flow) 
#         x = range(len(sensor)
        
        vid_qual_win_cnt, df_dataset, info_dataset = seg_smk_video(
#             subject=subject,
            video=video,
            flow = flow,
            sensor = sensor,
            window_size_sec=window_size_sec,
            stride_sec=stride_sec,
            offset_sec=offset_sec,
            kde_num_offset=kde_num_offset,
            kde_max_offset=kde_max_offset,
            window_criterion=window_criterion,
#             starttime_file=starttime_file,
            fps=fps,
        )

        vid_qual_win_cnt_all += vid_qual_win_cnt
        df_dataset_all += df_dataset
        info_dataset_all += info_dataset
        
        
    title_suffix = "_win{}_str{}_offset{}_rdoffset{}_maxoffset{}_wincrt{}".format(
        window_size_sec,
        stride_sec,
        offset_sec,
        kde_num_offset,
        kde_max_offset,
        window_criterion,
    )
    print(
        len(vid_qual_win_cnt_all),
        "videos with valid window(s), # of qualified windows: ",
        vid_qual_win_cnt_all,
    )
#     pd.DataFrame(vid_qual_win_cnt_all, columns=["vid_name", "window_num"]).to_csv(
#         "./data/num_valid_windows" + title_suffix + ".csv", index=None
#     )

    # with open(
    #     os.path.join(data_dir, "all_video" + title_suffix + "_df_dataset.pkl"), "wb"
    # ) as handle:
    #     pickle.dump(df_dataset_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(
    #     os.path.join(data_dir, "all_video" + title_suffix + "_info_dataset.pkl"), "wb"
    # ) as handle:
    #     pickle.dump(info_dataset_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return df_dataset_all, info_dataset_all

