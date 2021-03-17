import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import curve_fit
from textwrap import wrap
from tqdm import tqdm

def gaussian(x, mu, sig):
    """
    Gaussian kernel

    Args:
        x: independent variable
        mu: mean in Gaussian kernel
        sig: variance in Gaussian kernel

    Returns:
        Gaussian kernel function

    """
    return 1 / (2 * np.pi * sig) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def gaussian_voting(scores, kernel_var=500, draw=True, path='../figures/offset.jpg'):
    """
    Gaussian vote for a video

    Args:
        scores: numpy array, n x 2, each row consists of (conf, offset)
        kernel_var: int, variance kernel in Gaussian
        draw: boolean, draw figure or not
        path: str, directory to save figure

    Returns:
        float, offset
        float, conf
        list, [popt, pcov]

    """
    # INPUT: n x 2, conf, offset
    # OUTPUT: offset
    offset_max = 20000
    x = np.arange(-offset_max, offset_max + 1)
    y = np.zeros(2 * offset_max + 1)
    for i in range(scores.shape[0]):
        y += gaussian(x, scores[i, 1], kernel_var) * scores[i, 0]
    y /= np.sum(scores[:, 0])
    offset = np.argmax(y) - offset_max

    # fit a Gaussian to voted_shift using nonlinear least square
    # confidence of the shift estimation can be described as the variance of the estimated model parameters
    # conf = max(abs(y-median(y)))/stdev(y)
    try:
#         popt: array, Optimal values for the parameters so that the sum of the squared residuals 
#         of f(xdata, *popt) - ydata is minimized.
#         pcov: 2-D array The estimated covariance of popt. 
        popt, pcov = curve_fit(gaussian, x, y, bounds=([-offset_max, 0], [offset_max, np.inf]))
        y_nlm = gaussian(x, *popt)
    except RuntimeError:
        popt, pcov = np.array([np.inf, np.inf, np.inf]), \
                     np.array([[np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf]])
        y_nlm = np.zeros((len(x)))
    conf = 200000 / popt[1] / pcov[0, 0]
    if draw:
        plt.figure()
        plt.plot(x, y, color='blue', label='weighted kde')
        plt.plot(x, y_nlm, color='red', label='fitted gaussian')
        plt.xlabel('shift/ms')
        plt.ylabel('probability')
        plt.legend(loc='upper right')
        title = '{} windows, offset={}ms, conf={:.2f}'.format(scores.shape[0], int(offset), conf)
        plt.title("\n".join(wrap(title, 60)))
        plt.savefig(path)
        plt.close()

    return offset, conf, [popt, pcov]


def gaussian_voting_per_video(scores_dataframe, kernel_var=100, thresh=0, min_voting_segs=0, draw=True,
                              folder='../figures/cross_corr/'):
    """
    Calculate Gaussian voting result for a video

    Args:
        scores_dataframe: data frame, scores for each window in a video
        kernel_var: int, variance kernel in Gaussian
        thresh: float, threshold
        min_voting_segs: int, min of voting segments
        draw: boolean, draw figure or not
        folder: str

    Returns:
        dataframe, result df containing ['video', 'offset', 'abs_offset', 'num_segs', 'conf', 'mu', 'sigma', 'mu_var',
                                       'sigma_var', 'abs_mu']
        float, average segments

    """
    # INPUT: n x 3, conf, offset, video
    scores = scores_dataframe[['confidence', 'drift', 'video']].to_numpy()
#     print(scores.shape)
    scores = scores[scores[:, 0] > thresh]
    videos = np.unique(scores_dataframe[['video']].to_numpy())
    offset = np.zeros((len(videos)))
    conf = np.zeros((len(videos)))
    nlm_params = np.zeros((len(videos), 4))
    num_valid_segs = np.zeros((len(videos)))
    num_segs = 0
    num_videos = 0
    print("gaussian_voting_per_video")
    for i, vid in enumerate(tqdm(videos)):
#         print(i)
        path = os.path.join(folder, 'offset_' + vid)
        valid_segs = scores[:, 2] == vid
        num_segs_cur = sum(valid_segs)
        if num_segs_cur > min_voting_segs:
            offset[i], conf[i], p = gaussian_voting(scores[valid_segs, :2], kernel_var, draw, path)
            nlm_params[i, :] = np.concatenate((p[0][:2], np.diag(p[1])[:2]))
            num_valid_segs[i] = num_segs_cur
            num_segs += num_segs_cur
            num_videos += 1
        else:
            offset[i] = np.nan
            conf[i] = np.nan
    try:
        ave_segs = num_segs / num_videos
    except ZeroDivisionError:
        ave_segs = np.nan
    summary_df = pd.DataFrame(np.concatenate(
        [np.stack([videos, offset, abs(offset), num_valid_segs, conf], axis=1), nlm_params, abs(nlm_params[:, :1])],
        axis=1), \
                              columns=['video', 'offset', 'abs_offset', 'num_segs', 'conf', 'mu', 'sigma', 'mu_var',
                                       'sigma_var', 'abs_mu'])

    return summary_df, ave_segs
