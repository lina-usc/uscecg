import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from scipy import interpolate
import seaborn as sns
from pathlib import Path
import mne
import heartpy as hp

import warnings

def get_outliers(vals, k=3):
    q1, q2, q3 = np.percentile(vals, [25, 50, 75])
    return (vals < q2 - k * (q3 - q1)) | (vals > q2 + k * (q3 - q1))


def process_ecg_segment(file_name, debug=False):
    raw = mne.io.read_raw_edf(file_name, verbose=False)
    sfreq = raw.info["sfreq"]

    rec_ecg = raw.get_data().squeeze()

    return process_ecg_data_segment(rec_ecg, sfreq, debug=debug)


def process_ecg_data_segment(rec_ecg, sfreq, margin=250, debug=False, resample=None, **kwargs):
    if len(rec_ecg) == 0:
        return

    if resample is not None and resample != sfreq:
        rec_ecg = mne.filter.resample(rec_ecg, up=resample/sfreq)
        sfreq = resample

    try:
        wd, m = hp.process(rec_ecg, sfreq, **kwargs)
    except hp.exceptions.BadSignalWarning:
        return

    beats = []
    peaks, peaks_y = np.array([[peak, peak_y] for peak, peak_y in zip(wd["peaklist"], wd["ybeat"])
                               if peak not in wd["removed_beats"]]).T
    peaks = peaks.astype(int)

    p1s, p2s, p3s = peaks[:-2], peaks[1:-1], peaks[2:]
    for p1, p2, p3 in zip(p1s, p2s, p3s):
        x = np.arange(p1, p3 + 1)
        y = rec_ecg[p1:(p3 + 1)]
        f = interpolate.interp1d(x, y)

        xnew = np.concatenate((np.linspace(p1, p2, margin), np.linspace(p2, p3, margin)))
        ynew = f(xnew)  # use interpolation function returned by `interp1d`
        beats.append(ynew)

    beats = np.array(beats)

    if len(beats) == 0:
        return
    residuals = np.trapz((beats - beats.mean(0)) ** 2, axis=1)
    outliers = get_outliers(residuals, k=6)

    nb_samples = np.median((p3s - p1s)[~outliers])

    # Flagging as outliers P2 to close to the borders
    outliers |= ((p2s - nb_samples // 2).astype(int) < 0)
    outliers |= ((p2s + nb_samples // 2).astype(int) >= len(rec_ecg))

    additional_removed_beats = np.array(peaks)[np.concatenate([[True], outliers, [True]])]
    additional_removed_beats_y = np.array(peaks_y)[np.concatenate([[True], outliers, [True]])]

    clean_beats = beats[~outliers, :]

    raw_beats = np.array([rec_ecg[int(p2 - nb_samples // 2):int(p2 + nb_samples // 2)]
                          for p2 in p2s[~outliers]])

    raw_t = np.arange(-int(nb_samples // 2), int(nb_samples // 2)) / sfreq

    if clean_beats.shape[0] < 20:
        return

    wd_copy = wd.copy()

    wd_copy["removed_beats"] = np.array(np.concatenate([wd["removed_beats"], additional_removed_beats]))
    wd_copy["removed_beats_y"] = np.array(np.concatenate([wd["removed_beats_y"], additional_removed_beats_y]))

    clean_mean_beat = np.median(clean_beats, 0)

    signal = np.trapz(clean_mean_beat ** 2)
    noise = np.trapz((clean_beats - clean_mean_beat) ** 2, axis=1)

    if debug:
        plt.figure()
        hp.plotter(wd, m, figsize=(20, 4))
        plt.xlim(0, 30)

        plt.figure()
        plt.plot(residuals, ".")
        plt.plot(np.arange(len(residuals))[outliers], residuals[outliers], ".", color="r")

        plt.figure()
        hp.plotter(wd_copy, m, figsize=(20, 4))
        plt.xlim(0, 30)

        plt.figure()
        sns.heatmap(clean_beats)

        plt.figure()
        plt.plot(clean_beats.T, alpha=0.1, color='k')
        plt.plot(clean_mean_beat, color="r")

    return {"SNR": np.mean(10 * np.log10(signal / noise)),
            "mean_beat": clean_mean_beat,
            "nb_valid_beats": clean_beats.shape[0],
            "nb_invalid_beats": np.sum(outliers),
            # "file_parts": file_name.name.replace(".edf", "").split("_"),
            "wd": wd_copy,
            "clean_beats": clean_beats,
            "raw_beats": raw_beats,
            "raw_t": raw_t,
            "rel_p1": p2s[~outliers] - p1s[~outliers],
            "rel_p3": p3s[~outliers] - p2s[~outliers],
            "sfreq": sfreq}


def get_log_times(subject, age, path_timelog_format="Create_Segments/all_infants_timelogs/{subject}_{age}.csv",
                  datavyu_format="Generated Files_{kind}_03092022_Datavyu_ALLOnly_AI/Stimuli/" +
                                 "{subject}_{age}_stimulus.csv"):

    # Look first for datavyu times
    rows = []
    for kind in ["OIX", "PIX"]:
        stim_path = Path(datavyu_format.format(kind=kind, subject=subject, age=age))
        if stim_path.exists():
            csv_file = pd.read_csv(stim_path).dropna()
            csv_file.columns = ["start", "end", "stimulus"]
            csv_file = csv_file[csv_file.stimulus != "END"]
            rows.append({"start": csv_file.start.min() / 60.0,
                         "end": csv_file.end.max() / 60.0,
                         "condition": kind})
    if len(rows):
        return pd.DataFrame(rows)

    # if datavyu times are not available, look for old time logs
    path_timelog = Path(path_timelog_format.format(subject=subject, age=age))
    if path_timelog.exists():
        csv_file = pd.read_csv(path_timelog).dropna()
        csv_file.columns = ["visit", "segment", "condition", "start", "end"]
        csv_file = csv_file[csv_file.end > csv_file.start]
        return csv_file

    # No segment logs available
    return None


def get_segments(path_edf, **kwargs):

    subject, age = path_edf.name.replace(".edf", "").split("_")
    log_df = get_log_times(subject, age, **kwargs)
    if log_df is None:
        return None

    edf_raw = mne.io.read_raw_edf(path_edf, preload=True)
    sfreq = edf_raw.info["sfreq"]

    edf_raw = edf_raw.notch_filter(np.arange(60, sfreq/2.0, 60))
    edf_raw = edf_raw.filter(1, sfreq/4.0)

    try:
        starts = (log_df.start * 60 * sfreq).astype(int)
        stops = (log_df.end * 60 * sfreq).astype(int)
    except:
        print(log_df)
        raise

    # Reading each row start stop in excel file (timelogs)
    segments = []
    for start, stop, condition in zip(starts, stops, log_df.condition.values):
        if stop > len(edf_raw.times):
            warnings.warn(f"Condition {condition} for file {path_edf.name} stop at sample {stop}"
                          f" while the recording contains only {len(edf_raw.times)} samples.")
        segment = edf_raw.get_data("ECG0", start, stop).squeeze()
        if segment is not None and len(segment):
            segments.append(segment)
    return segments, log_df.condition.values, sfreq

