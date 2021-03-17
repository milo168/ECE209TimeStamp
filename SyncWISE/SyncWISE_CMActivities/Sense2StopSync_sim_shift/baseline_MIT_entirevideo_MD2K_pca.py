import os
import sys
import pickle
import numpy as np
import random
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), "../syncwise"))
from cross_correlation_func import cross_correlation_using_fft, compute_shift
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA

sys.path.append('src')
from utils import create_folder
from utils import csv_read
from resample import resample
from load_sensor_data import read_data_datefolder_hourfile
from settings import settings


CUBIC = 1 # Resampling method. 1: cubic spline, 0: linear interpolation

FRAME_INTERVAL = settings["FRAME_INTERVAL"]

startime_file = settings["STARTTIME_FILE"]

videos_df = pd.read_csv(os.path.join(startime_file))
len_all= len(videos_df)
len_val= int(round(len_all*settings["val_set_ratio"]))
random.seed(80)
# indices for validation set (sensitivity study)
val = random.sample(range(0, len_all), len_val)

videos_df = pd.read_csv(settings["STARTTIME_FILE"])
# save start time file for test videos
test_videos_df = videos_df.iloc[~videos_df.index.isin(val)]
test_videos_df = test_videos_df.sort_values("video_name")
test_videos_df.to_csv(settings["STARTTIME_TEST_FILE"], index=None)



def plot_MIT_baseline(df_resample, fps, out_path):
    fps=29.969664
    df_resample = df_resample.reset_index()
    df_resample['time'] -= df_resample['time'][0]
    df_resample['time'] /= 1000
    df_resample = df_resample.set_index('time')
    fig, axes = plt.subplots(4, 1, figsize=(20, 10))
    grid = plt.GridSpec(2, 3)
    axes[0].plot(df_resample['accx'],  label='accx')
    axes[0].plot(df_resample['accy'], label='accy')
    axes[0].plot(df_resample['accz'], label='accz')
    axes[0].legend(loc='upper right')
    axes[0].set_title('Accelerometer data')

    # plt.set_color_cycle(['red', 'black', 'blue', 'yellow', 'grey'])
    axes[1].plot(df_resample['diff_flowx'], 'b', label='diff_flowx')
    axes[2].plot(df_resample['diff_flowy'], 'orange', label='diff_flowy')
    #ax2.legend(loc='upper right')
    axes[1].set_title('diff flow x')
    axes[2].set_title('diff flow y')

    fftshift = cross_correlation_using_fft(df_resample['diff_flowx'].values, df_resample['accx'].values)
    shift = compute_shift(fftshift)
    axes[3].plot(fftshift)
    axes[3].set_title('diff flowx/accx delta={:.1f} ms'.format(shift * 1000/fps))
    
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(out_path)
    plt.close()
    
def baseline_MIT_video_MD2K(window_size_sec, stride_sec, num_offsets, max_offset, window_criterion, offset_sec = 0, plot=0, use_PCA=0):
    # #subjects = ['202', '205', '211', '235', '236', '238', '240', '243']
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # # df_start_time = pd.read_csv(settings['STARTTIME_TEST_FILE'])    
    # # df_start_time = csv_read(os.path.join(dir_path, '../../Sense2StopSync/start_time.csv')).set_index('video_name')
    # df_start_time = csv_read(settings['STARTTIME_TEST_FILE']).set_index('video_name')
    # video_names = df_start_time.index.tolist()
    # subjects = list(set([vid.split(' ')[0] for vid in video_names]))    
    # fps = 29.969664
    # data_dir = '/media/yun/08790233DP/sync_data_2nd'
    # with open(data_dir+'/all_video' + title_suffix + '_info_dataset.pkl', 'rb') as handle:
    #     info_dataset = pickle.load(handle)

    # video_all = []
    # for info in info_dataset:
    #     video_all.append(info[0])
    # counter = Counter(video_all)
    # print(counter)

    # # select the qualified videos with more than 20 windows
    # qualify_videos = []
    # # set the parameter number of qualified windows 
    # qualified_window_num = 200
    # for vid in counter:
    #     if counter[vid] > qualified_window_num:
    #         qualify_videos.append(vid)
    # print(len(qualify_videos), 'videos have more than ', qualified_window_num, ' qualified windows.\n')
    # print(qualify_videos)
    df_subjects = []
    suffix = 'num_offsets{}'.format(num_offsets) if num_offsets else ''
    title_suffix = '_win{}_str{}_offset{}_rdoffset{}_maxoffset{}_wincrt{}'.\
        format(window_size_sec, stride_sec, offset_sec, num_offsets, max_offset, window_criterion)
    data_dir = './'

    df_start_time = pd.read_csv(settings['STARTTIME_TEST_FILE'])
    qualify_videos  = df_start_time["video_name"].tolist()
    print(qualify_videos)
    print(len(qualify_videos))
    subjects = list(set([vid[:3] for vid in qualify_videos]))
    print(subjects)
    print(len(subjects))
    fps = settings['FPS']

    dir_path = './'

    DEVICE = 'CHEST'
    SENSOR = 'ACCELEROMETER'
    SENSORS = ['ACCELEROMETER_X', 'ACCELEROMETER_Y', 'ACCELEROMETER_Z']
    sensor_col_header = ['accx', 'accy', 'accz']
    start_time_file = settings['STARTTIME_FILE']
    RAW_PATH = settings['raw_path']
    RESAMPLE_PATH = settings['reliability_resample_path']

    create_folder("result/baseline_pca")

    for sub in subjects:
        DEVICE = 'CHEST'
        SENSOR = 'ACCELEROMETER'
        SENSORS = ['ACCELEROMETER_X', 'ACCELEROMETER_Y', 'ACCELEROMETER_Z']
        sensor_col_header = ['accx', 'accy', 'accz']

        # start_time_file = os.path.join(dir_path, '../../Sense2StopSync/start_time.csv')
        start_time_file = settings["STARTTIME_TEST_FILE"]

        #flow_dir = os.path.join(dir_path, '../../data/flow_pwc/sub{}'.format(sub))
        flow_dir = os.path.join(dir_path, '../../Sense2StopSync/flow_pwc/sub{}'.format(sub))
        #RAW_PATH = os.path.join(dir_path, '../../data/RAW/wild/')
        RAW_PATH = os.path.join(dir_path, '../../Sense2StopSync/RAW/wild/')
        #RESAMPLE_PATH = os.path.join(dir_path, '../../data/RESAMPLE200/wild/')
        RESAMPLE_PATH = os.path.join(dir_path, '../../Sense2StopSync/RESAMPLE200/wild/')

        flow_files = [f for f in os.listdir(flow_dir) if os.path.isfile(os.path.join(flow_dir, f))]
        flow_files = [f for f in flow_files if f.endswith('.pkl')]
        print('subject', sub, ': ', len(flow_files), 'total videos (including unqualified ones)')

        video_list = []
        offset_list = []

        for f in flow_files:
            vid_name = f[:-4]
            if vid_name not in qualify_videos:
                # print(vid_name, 'flow file not exist')
                continue
            vid_path = os.path.join(flow_dir, vid_name+'.pkl')
            out_path = os.path.join(data_dir, 'figures/figures_MIT_pca/corr_flow_averaged_acc_{}.png'.format(vid_name))
            
            
            # load start end time
            offset = 0
            df_start_time = pd.read_csv(start_time_file, index_col='video_name')
            if vid_name not in df_start_time.index:
                # print(vid_name, 'not exist in starttime csv')
                continue
            print(vid_name)
            start_time = df_start_time.loc[vid_name]['start_time']+offset
            
            # load optical flow data and assign unixtime to each frame
            motion = pickle.load(open(vid_path, 'rb'))
            # step = 1000.0/30.0
            step = 1000.0/fps
            length = motion.shape[0]
            timestamps_int = np.arange(start_time, start_time + length * step, step).astype(int)

            # # load sensor data
            # interval = [int(start_time), int(start_time) + length * step]
            # df = read_data_datefolder_hourfile(RESAMPLE_PATH, sub, DEVICE, SENSOR, *interval)
            # len_raw_sensor = len(df)

            # # load sensor reliability data
            # df_rel = read_data_datefolder_hourfile(RESAMPLE_PATH, sub, DEVICE, SENSOR + '_reliability', *interval)
            # # use the threshold ">=8Hz" to select 'good' seconds
            # rel_seconds = df_rel[df_rel['SampleCounts'] > 7].sort_values(by='Time')['Time'].values

            

            timestamps_int = timestamps_int[:min(len(timestamps_int), motion.shape[0])]
            motion = motion[:min(len(timestamps_int), motion.shape[0]), :]
            assert len(timestamps_int) == motion.shape[0]
            df_flow = pd.DataFrame({'time': timestamps_int, 'flowx': motion[:, 0], 'flowy': motion[:, 1]})
            df_flow['second'] = (df_flow['time']/1000).astype(int)

            # # extract the optical flow frames of the good seconds according to sensor data
            # df_flow_rel = pd.concat([df_flow[df_flow['second']==i] for i in rel_seconds]).reset_index()
            
            ## remove/keep video based on data quality
            # print(len(df_flow_rel)/len(df_flow))
            # if len(df_flow_rel)/len(df_flow) < 0.7:
                # continue

            fixedTimeCol = df_flow['time'].values

            if CUBIC:
                df_flow['time'] = pd.to_datetime(df_flow['time'], unit='ms')
            df_flow = df_flow[['flowx', 'flowy', 'time']].set_index('time')
            
            # extract the data of consecutive chunk and resample according to video frame timestamp
            df_list = []
            for S, col in zip(SENSORS, sensor_col_header):
                df = read_data_datefolder_hourfile(RAW_PATH, sub, DEVICE, S, fixedTimeCol[0], fixedTimeCol[-1])
                df = df[['time', col]]

                if CUBIC == 0:
                    df_sensor_resample = resample(df, 'time', samplingRate=0,
                                                  gapTolerance=200, fixedTimeColumn=fixedTimeCol).set_index('time')
                else:                        
                    df["time"] = pd.to_datetime(df["time"], unit="ms")
                    df = df.set_index("time")
                    df_sensor_resample = df.resample(FRAME_INTERVAL).mean()  
                    # FRAME_INTERVAL as 0.03336707S is the most closest value to 1/29.969664 pandas accepts
                    df_sensor_resample = df_sensor_resample.interpolate(method="spline", order=3) # cubic spline interpolation

                df_list.append(df_sensor_resample)
            
            if CUBIC == 0:
                df_list.append(df_flow)
                df_resample = pd.concat(df_list, axis=1)
            else:
                df_sensors = pd.concat(df_list, axis=1)
                df_list = [df_sensors, df_flow]
                df_resample = pd.merge_asof(df_list[1], df_list[0], on='time', tolerance=pd.Timedelta("30ms"), \
                direction='nearest').set_index('time')

            
            # two method to fill na values
            # 1 fill with 0
            #df_resample = df_resample.fillna(0)

            # 2 ffill
            df_resample = df_resample.fillna(method='ffill')
            df_resample = df_resample.dropna(how='any')

            df_resample['diff_flowx'] = df_resample['flowx'].diff()
            df_resample['diff_flowy'] = df_resample['flowy'].diff()
            df_resample = df_resample.dropna(how='any')
            pca_sensor = PCA(n_components=1)
            df_resample[['accx', 'accy', 'accz']] -= df_resample[['accx', 'accy', 'accz']].mean()
            df_resample['acc_pca'] = pca_sensor.fit_transform(df_resample[['accx', 'accy', 'accz']].to_numpy())
            diffflow_mat = df_resample[['diff_flowx', 'diff_flowy']].to_numpy()
            diffflow_mat -= np.mean(diffflow_mat, axis=0)
            pca_diffflow = PCA(n_components=1)
            df_resample['diffflow_pca'] = pca_diffflow.fit_transform(diffflow_mat)

            if use_PCA == 1:
                fftshift = cross_correlation_using_fft(df_resample['diffflow_pca'].values, df_resample['acc_pca'].values)
            else:
                fftshift = cross_correlation_using_fft(df_resample['diff_flowx'].values, df_resample['accx'].values)
            shift = compute_shift(fftshift)
            shift_ms = shift * 1000 / fps
            # print('diff_flowx accx delta={:.1f} ms'.format(shift_ms))
            video_list.append(vid_name)
            offset_list.append(shift_ms)
            print(vid_name, shift_ms)

            if plot:
                plot_MIT_baseline(df_resample, fps, out_path)

        df_subj = pd.DataFrame({'video': video_list, 'offset': offset_list})
        # print((df_subj))
        df_subjects.append(df_subj)
    result_df = pd.concat(df_subjects)
    result_df = result_df.reset_index()
    ave_error = np.mean(np.abs(result_df['offset'].values))
    PV300 = np.sum(np.abs(result_df['offset'].values) < 300) / len(result_df) * 100
    PV700 = np.sum(np.abs(result_df['offset'].values) < 700) / len(result_df) * 100
    result_df.to_csv(os.path.join(data_dir, 'result/baseline_pca/baseline_MIT_entirevideo_MD2K_offset_pad' + title_suffix + '.csv'), index=None)
    print("ave_error", "PV300", "PV700")
    print(ave_error, PV300, PV700)
    print("# videos: ", len(result_df))
    return result_df

def summarize_batch_result():
    file_path = './file_list_random.txt'
    summ_path = './result/batch_result_baseline_MIT_random.csv'
    stride_sec = 1
    num_offsets = 20
    with open(file_path) as f:
        params = f.readlines()
    #params = params[:1]
    fout = open(summ_path, 'w')
    fout.write('window_size_sec, stride_sec, num_offsets, window_criterion, max_offset, num_videos, ave_offset, num_1000ms, num_700ms, num_300ms\n')
    for line in params:
        window_size_sec, window_criterion, max_offset, num_offsets, offset_sec = line.split()
        title_suffix = '_win{}_str{}_rdoffset{}_maxoffset{}_wincrt{}_pca'\
            .format(window_size_sec, stride_sec, num_offsets, max_offset, window_criterion)
        result_df = baseline_MIT_video_MD2K(window_size_sec, stride_sec, num_offsets, max_offset, window_criterion, offset_sec=offset_sec)
        print(result_df)
        error_abs = abs(result_df['offset'].to_numpy())
        ave_offset = np.mean(error_abs[~np.isnan(error_abs.astype(np.float))])
        num_1000ms = len(error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 1000)])
        num_700ms = len(error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 700)])
        num_300ms = len(error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 300)])
        num_videos = len(result_df)
        assert len(error_abs) == len(result_df)
        fout.write(','.join(map(str, [window_size_sec, stride_sec, num_offsets, window_criterion, max_offset, num_videos, ave_offset, num_1000ms, num_700ms, num_300ms]))+'\n')
    fout.close()


if __name__ == "__main__":
    window_size_sec = 10
    stride_sec = 1
    num_offsets = 20
    max_offset = 5000
    window_criterion = 0.8
    plot = 0
    offset_sec = 1.9898898767622777

    # parameter_str = sys.argv[1]
    # window_size_sec, window_criterion, kde_max_offset, kde_num_offset, offset_sec = parameter_str.split(' ')
    # window_size_sec = int(window_size_sec)
    # window_criterion = float(window_criterion)
    # kde_max_offset = int(kde_max_offset)
    # kde_num_offset = int(kde_num_offset)
    # if float(offset_sec).is_integer():
    #     offset_secs = [int(offset_sec)]
    # else:
    #     offset_secs = [float(offset_sec)]
    # stride_sec = 1
    use_PCA = 1
    print("if use PCA or use x-axis: (1: PCA; 0: x-axis)", use_PCA)

    result_df = baseline_MIT_video_MD2K(window_size_sec, stride_sec, num_offsets, max_offset, window_criterion, offset_sec=offset_sec, plot=plot, use_PCA=use_PCA)
    print(result_df)
    # summarize_batch_result()


    use_PCA = 0
    print("if use PCA or use x-axis: (1: PCA; 0: x-axis)", use_PCA)

    result_df = baseline_MIT_video_MD2K(window_size_sec, stride_sec, num_offsets, max_offset, window_criterion, offset_sec=offset_sec, plot=plot, use_PCA=use_PCA)
    print(result_df)
    # summarize_batch_result()


