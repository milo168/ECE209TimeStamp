import numpy as np
import pandas as pd

from settings import settings

# =====================================================================================
FPS = settings["FPS"]
FRAME_INTERVAL = settings["FRAME_INTERVAL"]
# STARTTIME_FILE = settings['STARTTIME_TEST_FILE']
reliability_resample_path = settings['reliability_resample_path']
raw_path = settings['raw_path']
flow_path = settings['flow_path']
stride_sec = settings['STRIDE_SEC']
kde_max_offset = settings['kde_max_offset']
# =====================================================================================

def read_batch_final_results(
    window_size_sec,
    stride_sec,
    offset_sec,
    kde_num_offset,
    kde_max_offset,
    window_criterion,
    folder='./result/',
):
    """
    Read batch final results.

    Args:
        window_size_sec: int, window size
        stride_sec: int, stride
        offset_sec: float, offset in seconds
        kde_num_offset: int, KDE algorithm number of offset
        kde_max_offset: int, KDE algorithm max offset
        window_criterion: float, window criterion
        folder: str

    Returns:
        float, ave offset
        float, PV1000
        float, PV700
        float, PV300
        float, ave_conf
        int, num of videos
        int, ave_num_segments
        float, confidence

    """
    title_suffix = "_win{}_str{}_offset{}_rdoffset{}_maxoffset{}_wincrt{}".format(
        window_size_sec,
        stride_sec,
        offset_sec,
        kde_num_offset,
        kde_max_offset,
        window_criterion,
    )
    result_file = folder + "/final_result_per_video" + title_suffix + ".csv"
    result_df = pd.read_csv(result_file)
    num_videos = len(result_df)
    error_abs = abs((result_df["offset"] + offset_sec * 1000).to_numpy())
    num_segs = result_df["num_segs"].to_numpy()
    try:
        ave_offset = np.mean(error_abs[~np.isnan(error_abs.astype(np.float))])
    except ZeroDivisionError:
        ave_offset = 0
        print("Error @offset_sec:", offset_sec, "kde_max_offset", kde_max_offset)
    try:
        ave_num_segs = np.mean(num_segs[~np.isnan(num_segs.astype(np.float))])
        PV1000 = len(
            error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 1000)]
        ) / float(num_videos)
        PV700 = len(
            error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 700)]
        ) / float(num_videos)
        PV300 = len(
            error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 300)]
        ) / float(num_videos)
        conf = 200000 / (result_df["sigma"].to_numpy() * result_df["mu_var"].to_numpy())
        ave_conf = np.mean(conf[~np.isnan(conf.astype(np.float))])
        return (
            ave_offset,
            PV1000,
            PV700,
            PV300,
            ave_conf,
            num_videos,
            ave_num_segs,
            conf,
        )
    except:
        print(title_suffix)
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def summarize_xaxis_batch_to_csv(summ_path):
    """
    summarize xaxis batch and save to csv

    Args:
        summ_path: str, summary path

    Returns:
        None

    """
    file_path = "./file_list_random.txt"
    stride_sec = 1
    with open(file_path) as f:
        params = f.readlines()
    fout = open(summ_path, "w")
    fout.write(
        "window_size_sec,stride_sec,window_criterion,max_offset,kde_num_offset,offset_sec,num_videos,ave_num_segs,ave_offset,ave_conf,PV1000,PV700,PV300\n"
    )
    for line in params:
        (
            window_size_sec,
            window_criterion,
            max_offset,
            kde_num_offset,
            offset_sec,
        ) = line.split()
        title_suffix = "_win{}_str{}_offset{}_rdoffset{}_maxoffset{}_wincrt{}".format(
            window_size_sec,
            stride_sec,
            offset_sec,
            kde_num_offset,
            max_offset,
            window_criterion,
        )
        result_file = "result/summary_xx/final_result_per_video" + title_suffix + ".csv"
        result_df = pd.read_csv(result_file)
        num_videos = len(result_df)
        error_abs = abs((result_df["offset"] + float(offset_sec) * 1000).to_numpy())
        ave_offset = np.mean(error_abs[~np.isnan(error_abs.astype(np.float))])
        PV1000 = len(
            error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 1000)]
        ) / float(num_videos)
        PV700 = len(
            error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 700)]
        ) / float(num_videos)
        PV300 = len(
            error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 300)]
        ) / float(num_videos)
        conf = 200000 / (result_df["sigma"].to_numpy() * result_df["mu_var"].to_numpy())
        ave_conf = np.mean(
            conf[~np.isnan(conf.astype(np.float)) & (conf < 10000)]
        )  # dive into the problem there are two >10000 cases
        assert len(error_abs) == len(result_df)
        ave_num_segs = np.mean(result_df["num_segs"].to_numpy())
        fout.write(
            ",".join(
                map(
                    str,
                    [
                        window_size_sec,
                        stride_sec,
                        window_criterion,
                        max_offset,
                        kde_num_offset,
                        offset_sec,
                        num_videos,
                        ave_num_segs,
                        ave_offset,
                        ave_conf,
                        PV1000,
                        PV700,
                        PV300,
                    ],
                )
            )
            + "\n"
        )
    fout.close()
    
    result_df = pd.read_csv(summ_path)
    with open('final/syncwise_xaxis_final_result.txt', 'w') as f:
        print("Ave #Win Pairs: ", result_df["ave_num_segs"].mean(), file=f)
        print("Ave Error (ms): ", result_df["ave_offset"].mean(), file=f)
        print("PV700 (%): ", result_df["PV700"].mean(), file=f)
        print("PV300 (%): ", result_df["PV300"].mean(), file=f)

    print("Ave #Win Pairs: ", result_df["ave_num_segs"].mean())
    print("Ave Error (ms): ", result_df["ave_offset"].mean())
    print("PV700 (%): ", result_df["PV700"].mean())
    print("PV300 (%): ", result_df["PV300"].mean())


def summarize_pca_batch_to_csv(summ_path):
    """
    summarize pca batch and save to csv

    Args:
        summ_path: str, summary path

    Returns:
        None

    """
    file_path = "./file_list_random.txt"
#     stride_sec = 1
    with open(file_path) as f:
        params = f.readlines()
    fout = open(summ_path, "w")
    fout.write(
        "window_size_sec,stride_sec,window_criterion,max_offset,kde_num_offset,offset_sec,num_videos,ave_num_segs,ave_offset,ave_conf,PV1000,PV700,PV300,PV100\n"
    )
    for line in params:
        (
            window_size_sec,
            window_criterion,
            max_offset,
            kde_num_offset,
            offset_sec,
        ) = line.split()
        # imort from setting
        window_size_sec = settings["window_size_sec"]
        offset_sec = settings["offset_sec"]
        kde_num_offset = settings["kde_num_offset"]
        max_offset = settings["kde_max_offset"]
        window_criterion = settings["window_criterion"]
        
        title_suffix = "_win{}_str{}_offset{}_rdoffset{}_maxoffset{}_wincrt{}".format(
            float(window_size_sec),
            stride_sec,
            offset_sec,
            kde_num_offset,
            max_offset,
            window_criterion,
        )
#         print(str(title_suffix))
        result_file = (
            "result/summary_pca/final_result_per_video" + title_suffix + ".csv"
        )
        print('summarize read:', result_file)
        try:
            result_df = pd.read_csv(result_file)
        except:
            continue
        num_videos = len(result_df)
        error_abs = abs((result_df["offset"] + float(offset_sec) * 1000).to_numpy())
        ave_offset = np.mean(error_abs[~np.isnan(error_abs.astype(np.float))])
        PV1000 = len(
            error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 1000)]
        ) / float(num_videos)
        PV700 = len(
            error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 700)]
        ) / float(num_videos)
        PV300 = len(
            error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 300)]
        ) / float(num_videos)
        PV100 = len(
            error_abs[(~np.isnan(error_abs.astype(np.float))) & (error_abs <= 100)]
        ) / float(num_videos)
        conf = 200000 / (result_df["sigma"].to_numpy() * result_df["mu_var"].to_numpy())
        ave_conf = np.mean(
            conf[~np.isnan(conf.astype(np.float)) & (conf < 10000)]
        )  # dive into the problem there are two >10000 cases
        assert len(error_abs) == len(result_df)
        ave_num_segs = np.mean(result_df["num_segs"].to_numpy())
        fout.write(
            ",".join(
                map(
                    str,
                    [
                        window_size_sec,
                        stride_sec,
                        window_criterion,
                        max_offset,
                        kde_num_offset,
                        offset_sec,
                        num_videos,
                        ave_num_segs,
                        ave_offset,
                        ave_conf,
                        PV1000,
                        PV700,
                        PV300,
                        PV100,
                    ],
                )
            )
            + "\n"
        )
    fout.close()

    result_df = pd.read_csv(summ_path)
#     print(str(summ_path))
    with open("final/syncwise_pca_final_result.txt", 'w') as f:
        print("Ave #Win Pairs: ", result_df["ave_num_segs"].mean(), file=f)
        print("Ave Error (ms): ", result_df["ave_offset"].mean(), file=f)
        print("PV700 (%): ", result_df["PV700"].mean(), file=f)
        print("PV300 (%): ", result_df["PV300"].mean(), file=f)
        print("PV100 (%): ", result_df["PV100"].mean(), file=f)

    print("Ave #Win Pairs: ", result_df["ave_num_segs"].mean())
    print("Ave Error (ms): ", result_df["ave_offset"].mean())
    print("PV700 (%): ", result_df["PV700"].mean())
    print("PV300 (%): ", result_df["PV300"].mean())
    print("PV100 (%): ", result_df["PV100"].mean())


if __name__ == "__main__":
#     summarize_xaxis_batch_to_csv('./result/batch_result_xx_sigma500_flow_w_random.csv')
    summarize_pca_batch_to_csv('./result/batch_result_pca_sigma500_flow_w_random.csv')
