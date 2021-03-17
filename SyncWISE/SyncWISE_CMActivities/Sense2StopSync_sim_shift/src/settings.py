import os
import pytz


settings = {}
settings["TEMP_DIR"] = "tmp_data"
settings["TIMEZONE"] = pytz.timezone("America/Chicago")
# settings["FPS"] = 29.969664
settings["FPS"] = 29.0
# settings["FRAME_INTERVAL"] = "0.03336707S"
settings["FRAME_INTERVAL"] = "0.03448275S" 

# settings["STARTTIME_FILE"] = "start_time.csv"
# settings["STARTTIME_VAL_FILE"] = "start_time_val.csv"
# settings["STARTTIME_TEST_FILE"] = "start_time_test.csv"


DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../Sense2StopSync/")
settings["DATA_DIR"] = DATA_DIR
settings["reliability_resample_path"] = os.path.join(DATA_DIR, "RESAMPLE/wild/")
settings["raw_path"] = os.path.join(DATA_DIR, "RAW/wild/")
settings["flow_path"] = os.path.join(DATA_DIR, "flow_pwc/")
settings["qualified_window_num"] = 2 # not used in fact
# select the qualified videos with a sufficient number of windows: > settings["qualified_window_num"] * kde_num_offset
settings["STRIDE_SEC"] = 0.300 #change:int->float
settings["kernel_var"] = 500 # 500 how to modify this?
settings["sample_counts"] = 8 #threshold
settings["video_max_len"] = (17*60+43)*1000
settings["val_set_ratio"] = 0.2
settings["test_videos_num"] = 1376 #1377 total for 0  shift; 1376 for others
settings["IMU_SHIFT"] = 40 # read the N shifted samples IMU data

# now chage here (change from file_list_random.txt
settings["window_size_sec"] = 0.500 # num_win: ((1.5-winsize)/stride+1)*kde_num_offset ;float, >=STRIDE_SEC; 
settings["window_criterion"] = 0.9 # drop row/column has any NA values. control window quality of pair {crop_flow, shift_IMU}
settings["kde_max_offset"] = 1000 # ms, change from file_list_random.txt
settings["kde_num_offset"] = 8 # change from file_list_random.txt
settings["offset_sec"] = 0.0

