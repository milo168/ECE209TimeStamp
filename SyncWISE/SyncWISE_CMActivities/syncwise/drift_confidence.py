from matplotlib import pyplot as plt
from cross_correlation_func import cross_correlation_using_fft, compute_shift
from scipy.stats import kurtosis
from statistics import mean, stdev, median
from textwrap import wrap


def drift_confidence(df_resample, out_path, fps, pca=1, save_fig=0):
    """
    calculate drift and confidence from sensor data and flow data

    Args:
        df_resample: dataframe, resampling dataframe
        out_path: str
        fps: float
        pca: boolean, use pca or not
        save_fig: boolean, save figure or not

    Returns:
        float, drift
        float, confidence

    """
    if pca:
        flow_key = 'diffflow_pca'
        acc_key = 'acc_pca'
    else:
        flow_key = 'diff_flowx'
        acc_key = 'accx'

    fftshift = cross_correlation_using_fft(df_resample[flow_key].values, df_resample[acc_key].values)
    dist = max(abs(fftshift-median(fftshift)))
    shift = compute_shift(fftshift)
#     print('shift', shift)
    fx_ay_drift = shift * 1000/fps
    fx_ay_conf = dist/stdev(fftshift)

    if save_fig:
        fig, ax = plt.subplots(2, 4, figsize=(20, 10))
        plt.subplot(2, 4, 1)
        plt.plot(df_resample['accx'])
        plt.plot(df_resample['accy'])
        plt.plot(df_resample['accz'])
        plt.title('acc x, y, z')
         
        plt.subplot(2, 4, 5)
        plt.plot(df_resample['diff_flowx'])
        plt.plot(df_resample['diff_flowy'])
        plt.title('diff_flow x & y')

        plt.subplot(2, 4, 2)
        fftshift = cross_correlation_using_fft(df_resample['diff_flowx'].values, df_resample['diff_flowy'].values)
        dist = max(abs(fftshift-median(fftshift)))
        shift = compute_shift(fftshift)
        plt.plot(fftshift)
        plt.title("\n".join(wrap('fx fy {:.1f} ms, k{:.1f}, std{:.1f}, dm{:.1f}, ndm{:.1f}'.format(\
                shift * 1000/fps, kurtosis(fftshift), stdev(fftshift), dist, dist/stdev(fftshift)), 40)))
        
        plt.subplot(2, 4, 3)
        fftshift = cross_correlation_using_fft(df_resample['diff_flowsquare'].values, df_resample['accsquare'].values)
        dist = max(abs(fftshift-median(fftshift)))
        shift = compute_shift(fftshift)
        plt.plot(fftshift)
        plt.title("\n".join(wrap('fsq asq {:.1f} ms, k{:.1f}, std{:.1f}, dm{:.1f}, ndm{:.1f}'.format(\
                shift * 1000/fps, kurtosis(fftshift), stdev(fftshift), dist, dist/stdev(fftshift)), 40)))
         
        plt.subplot(2, 4, 4)
        fftshift = cross_correlation_using_fft(df_resample['diff_flowx'].values, df_resample['accx'].values)
        dist = max(abs(fftshift-median(fftshift)))
        shift = compute_shift(fftshift)
        plt.plot(fftshift)
        plt.title("\n".join(wrap('fx ax {:.1f} ms, k{:.1f}, std{:.1f}, dm{:.1f}, ndm{:.1f}'.format(\
                shift * 1000/fps, kurtosis(fftshift), stdev(fftshift), dist, dist/stdev(fftshift)), 40)))
         
        plt.subplot(2, 4, 6)
        fftshift = cross_correlation_using_fft(df_resample['diff_flowy'].values, df_resample['accz'].values)
        dist = max(abs(fftshift-median(fftshift)))
        shift = compute_shift(fftshift)
        plt.plot(fftshift)
        plt.title("\n".join(wrap('fy az {:.1f} ms, k{:.1f}, std{:.1f}, dm{:.1f}, ndm{:.1f}'.format(\
                shift * 1000/fps, kurtosis(fftshift), stdev(fftshift), dist, dist/stdev(fftshift)), 40)))
         
        plt.subplot(2, 4, 7)
        fftshift = cross_correlation_using_fft(df_resample['diff_flowx'].values, df_resample['accy'].values)
        dist = max(abs(fftshift-median(fftshift)))
        shift = compute_shift(fftshift)
        plt.plot(fftshift)
        plt.title("\n".join(wrap(r'fx ay $\bf{{{:.1f}}}$ ms, k{:.1f}, std{:.1f}, dm{:.1f}, ndm$\bf{{{:.1f}}}$'.format(\
                shift * 1000/fps, kurtosis(fftshift), stdev(fftshift), dist, dist/stdev(fftshift)), 40)))

        fig.tight_layout()
        fig.subplots_adjust(top=0.8)
        plt.savefig(out_path)
        plt.close()    

    return fx_ay_drift, fx_ay_conf
