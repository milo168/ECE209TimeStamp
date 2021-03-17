import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import pickle

from settings import settings
from utils import create_folder

sys.path.append(os.path.join(os.path.dirname(__file__), "../../syncwise"))
from abs_error_ROC import gaussian_voting_per_video
from drift_confidence import drift_confidence


FPS = settings["FPS"]
KERNEL_VAR = settings["kernel_var"]
temp_dir = settings["TEMP_DIR"]

def calc_win_offset_all(
        df_dataset_all,
        info_dataset_all,
        window_size_sec,
        stride_sec,
        kde_num_offset,
        qualified_window_num,
        pca,
        fps,
        draw=0
):
    """
    Calculate window offset for all videos

    Args:
        df_dataset_all: dataframe, all videos dataset
        info_dataset_all: dataframe, all videos information
        window_size_sec: int, window size
        stride_sec: int, stride size
        kde_num_offset: int, in KDE algorithm number of offsets
        qualified_window_num: int, number of qualified windows
        pca: boolean, use pca or not
        fps: float
        draw: draw flag, default = 0

    Returns:
        dataframe, the starttime, offset and confidence for each window in video

    """
    video_all = []
    for info in info_dataset_all:
        video_all.append(info[0])
    counter = Counter(video_all)
    # select the qualified videos with a sufficient number of windows
    qualify_videos = []
    for i in counter:
        if counter[i] > qualified_window_num:
            qualify_videos.append(i)

    all_offset_list = []
    all_drift_list = []
    all_conf_list = []
    all_video_list = []
    all_starttime_list = []

    if draw and not os.path.exists("figures/MD2K_cross_corr_win" + str(window_size_sec) + "_str" + str(stride_sec)):
        os.makedirs("figures/MD2K_cross_corr_win" + str(window_size_sec) + "_str" + str(stride_sec))
    # iterate through all the qualified videos
    for video in qualify_videos:
        # for all the qualified windows in this video:
        for df, info in zip(df_dataset_all, info_dataset_all):
            if info[0] == video:
                out_path = (
                        "figures/MD2K_cross_corr_win"
                        + str(window_size_sec)
                        + "_str"
                        + str(stride_sec)
                        + "/corr_flow_acc_{}_{}".format(video, info[1])
                )
                drift, conf = drift_confidence(df, out_path, fps, pca=pca, save_fig=0)
                
                if kde_num_offset:
                    all_drift_list.append(drift - info[3])
                else:
                    all_drift_list.append(drift)

                all_offset_list.append(info[3])
#                 print('info[3]',info[3])
                all_conf_list.append(conf)
                all_video_list.append(video)
                all_starttime_list.append(info[1])

    df = pd.DataFrame(
        {
            "confidence": all_conf_list,
            "offset": all_offset_list,
            "drift": all_drift_list,
            "video": all_video_list,
            "starttime": all_starttime_list,
        }
    )
    return df


def print_offset_summary(offset_df):
    """
    Print summary for offset result

    Args:
        offset_df: dataframe, offset

    Returns:
        None

    """
    l = len(offset_df)
    l1 = len(offset_df[(offset_df["offset"] > -700) & (offset_df["offset"] < 700)])
    l2 = len(offset_df[(offset_df["offset"] > -300) & (offset_df["offset"] < 300)])
    offset_abs = abs(offset_df["offset"].to_numpy())
    ave_offset = np.mean(offset_abs[~np.isnan(offset_abs.astype(np.float))])
    ave_segs = np.mean(offset_df["num_segs"])
    print(
        "{}/{} videos with error < 700 ms, {}/{} videos with error time < 300 ms".format(
            l1, l, l2, l
        )
    )
    print("average offset = ", ave_offset)
    print("ave segs = ", ave_segs)


def calc_video_offset_all(
        df_dataset_all,
        info_dataset_all,
        window_size_sec,
        stride_sec,
        offset_sec,
        kde_num_offset,
        kde_max_offset,
        window_criterion,
        qualified_window_num,
        save_dir,
        pca,
        kernel_var=KERNEL_VAR,
        fps=FPS,
        draw=0
):
    """
    calculate video offset for all videos

    Args:
        df_dataset_all: dataframe, data for all videos
        info_dataset_all: dataframe, information for all videos
        window_size_sec: int,  window size
        stride_sec: int, stride
        offset_sec: float, offset
        kde_num_offset: int, KDE number of offset
        kde_max_offset: int, KDE max offset
        window_criterion: float, window criterion
        qualified_window_num: int, number of qualified window
        save_dir: str, save directory
        pca: boolean, use pca or not
        kernel_var: int, kernel variance
        fps: float
        draw: boolean, draw figure or not

    Returns:
        None
    """
    title_suffix = "_win{}_str{}_offset{}_rdoffset{}_maxoffset{}_wincrt{}".format(
        window_size_sec,
        stride_sec,
        offset_sec,
        kde_num_offset,
        kde_max_offset,
        window_criterion,
    )
    print('\n ground truth offset (ms)', offset_sec*1000, '\n')
    print('computing offset of each window')
    scores_dataframe = calc_win_offset_all(
        df_dataset_all=df_dataset_all,
        info_dataset_all=info_dataset_all,
        window_size_sec=window_size_sec,
        stride_sec=stride_sec,
        kde_num_offset=kde_num_offset,
        qualified_window_num=qualified_window_num,
        pca=pca,
        fps=fps,
        draw=0
    )
    print('   done')
    
    
#     scores_dataframe.to_csv(save_dir + '/conf_offset_win' + title_suffix + '.csv', index=None)

    if draw:
        create_folder("figures/MD2K_cross_corr" + title_suffix)

    offset_df, _ = gaussian_voting_per_video(
        scores_dataframe,
        kernel_var=kernel_var,
        thresh=0,
        draw=draw,
        folder="figures/MD2K_cross_corr" + title_suffix,
    )
    
#     # save offsets to .pkl
    offsets_np  = offset_df['offset'].to_numpy()
#     fileObject = open(save_dir + '/offsets.pkl', 'wb')
#     pickle.dump(offsets_np, fileObject)
#     fileObject.close()
    # save offsets to .npz
    path_sample=save_dir+'/offsets/'+'IMU_offsets_'+str(settings["IMU_SHIFT"])+'_shift'
    np.savez(path_sample, offsets_np)
#     print('offset:',offsets_np,'Saved in'+save_dir+'/')
    print('offsets: Saved in: '+save_dir+'/offsets/')
#     tmp = pickle.load(open(save_dir + "/offsets.pkl", "rb"))
#     print(tmp)
    
    offset_df = offset_df.sort_values(by=["offset"])
    offset_df.to_csv(
        save_dir + "/final_result_per_video" + title_suffix + ".csv", index=None
    )
    print('CSV saved in:', save_dir + "/final_result_per_video" + title_suffix + ".csv")
