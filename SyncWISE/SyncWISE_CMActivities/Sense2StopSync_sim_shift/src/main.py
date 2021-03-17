import sys

from segment_video_all import segment_video_all
from calc_video_offset_all import calc_video_offset_all
from settings import settings

import random

if __name__ == "__main__":
#     starttime_file = settings['STARTTIME_TEST_FILE']
    stride_sec = settings["STRIDE_SEC"]
    random.seed(0)

    parameter_str = sys.argv[1] # read form file
    (
        window_size_sec,
        window_criterion,  # criterion of sensor data quality for qualified window
        kde_max_offset,  # max kde offset (in milliseconds), kde: kernel density estimation
        kde_num_offset,  # number of kde offset
        offset_sec
    ) = parameter_str.split(" ")
#     print(parameter_str)
#     window_size_sec = int(window_size_sec)
#     window_size_sec = float(window_size_sec)
#     offset_sec = float(offset_sec)
#     kde_num_offset = int(kde_num_offset)
#     kde_max_offset = int(kde_max_offset)
#     window_criterion = float(window_criterion)
    window_size_sec = settings["window_size_sec"]
    offset_sec = settings["offset_sec"]
    kde_num_offset = settings["kde_num_offset"]
    kde_max_offset = settings["kde_max_offset"]
    window_criterion = settings["window_criterion"]

    # segment videos into windows
    df_dataset_all, info_dataset_all = segment_video_all(
        window_size_sec,
        stride_sec,
        offset_sec,
        kde_num_offset,
        kde_max_offset,
        window_criterion,
        # data_dir,
#         starttime_file,
    )

    # calculate drift for all the windows using PCA
    calc_video_offset_all(
        df_dataset_all,
        info_dataset_all,
        window_size_sec,
        stride_sec,
        offset_sec,
        kde_num_offset,
        kde_max_offset,
        window_criterion,
        qualified_window_num=settings["qualified_window_num"] * kde_num_offset,
        save_dir='./result/summary_pca',
        pca=1  # 0: no pca; 1: pca
    )

#     # calculate drift for all the windows using X-axis
#     calc_video_offset_all(
#         df_dataset_all,
#         info_dataset_all,
#         window_size_sec,
#         stride_sec,
#         offset_sec,
#         kde_num_offset,
#         kde_max_offset,
#         window_criterion,
#         qualified_window_num=settings["qualified_window_num"] * kde_num_offset,
#         save_dir='./result/summary_xx',
#         pca=0  # 0: no pca; 1: pca
#     )
