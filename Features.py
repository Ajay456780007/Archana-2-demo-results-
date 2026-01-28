# ---- Import Necessary python Modules ----
import os
from typing import List
from scipy.stats import skew, kurtosis
from termcolor import cprint
from HHT.torchHHT import hht
from sklearn.decomposition import PCA
import fathon
from fathon import fathonUtils as fu
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
from glob import glob
import scipy.io
import warnings
from keras.applications import VGG16
from keras.models import Model
import random
from scipy.signal import welch
import librosa
import numpy as np
import scipy
import scipy.signal
import scipy.fftpack
from tqdm import tqdm
import wfdb
from ssqueezepy import cwt, Wavelet
import cv2
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pywt
from scipy.signal import hilbert

warnings.filterwarnings("ignore")
from Extract_PQRST.Extract_PQRST import find_pqrst_peaks1, split_peaks


# ----------------- feature extraction -------------


def HRV_Features(nn_intervals: List[float], pnni_as_percent: bool = True):
    """
    Notes
    -----
    Here are some details about feature engineering...
    - **mean_nni**: The mean of RR-intervals.
    - **sdnn** : The standard deviation of the time interval between successive normal heart beats \
    (i.e. the RR-intervals).
    - **sdsd**: The standard deviation of differences between adjacent RR-intervals
    - **rmssd**: The square root of the mean of the sum of the squares of differences between \
    adjacent NN-intervals. Reflects high frequency (fast or parasympathetic) influences on hrV \
    (*i.e.*, those influencing larger changes from one beat to the next).
    - **median_nni**: Median Absolute values of the successive differences between the RR-intervals.
    - **nni_50**: Number of interval differences of successive RR-intervals greater than 50 ms.
    - **pnni_50**: The proportion derived by dividing nni_50 (The number of interval differences \
    of successive RR-intervals greater than 50 ms) by the total number of RR-intervals.
    - **nni_20**: Number of interval differences of successive RR-intervals greater than 20 ms.
    - **pnni_20**: The proportion derived by dividing nni_20 (The number of interval differences \
    of successive RR-intervals greater than 20 ms) by the total number of RR-intervals.
    - **range_nni**: difference between the maximum and minimum nn_interval.
    - **cvsd**: Coefficient of variation of successive differences equal to the rmssd divided by \
    mean_nni.
    - **cvnni**: Coefficient of variation equal to the ratio of sdnn divided by mean_nni.
    - **mean_hr**: The mean Heart Rate.
    - **max_hr**: Max heart rate.
    - **min_hr**: Min heart rate.
    - **std_hr**: Standard deviation of heart rate.
    """
    diff_nni = np.diff(nn_intervals)
    length_int = len(nn_intervals) - 1 if pnni_as_percent else len(nn_intervals)
    # Basic statistics
    mean_nni = np.mean(nn_intervals)
    median_nni = np.median(nn_intervals)
    range_nni = max(nn_intervals) - min(nn_intervals)

    sdsd = np.std(diff_nni)
    rmssd = np.sqrt(np.mean(diff_nni ** 2))

    nni_50 = sum(np.abs(diff_nni) > 50)
    pnni_50 = 100 * nni_50 / length_int
    nni_20 = sum(np.abs(diff_nni) > 20)
    pnni_20 = 100 * nni_20 / length_int

    # Feature found on github and not in documentation
    cvsd = rmssd / mean_nni

    # Features only for long term recordings
    sdnn = np.std(nn_intervals, ddof=1)  # ddof = 1 : unbiased estimator => divide std by n-1
    cvnni = sdnn / mean_nni

    # Heart Rate equivalent features
    heart_rate_list = np.divide(60000, nn_intervals)
    mean_hr = np.mean(heart_rate_list)
    min_hr = min(heart_rate_list)
    max_hr = max(heart_rate_list)
    std_hr = np.std(heart_rate_list)

    time_domain_features = {
        'mean_nni': mean_nni,
        'sdnn': sdnn,
        'sdsd': sdsd,
        'nni_50': nni_50,
        'pnni_50': pnni_50,
        'nni_20': nni_20,
        'pnni_20': pnni_20,
        'rmssd': rmssd,
        'median_nni': median_nni,
        'range_nni': range_nni,
        'cvsd': cvsd,
        'cvnni': cvnni,
        'mean_hr': mean_hr,
        "max_hr": max_hr,
        "min_hr": min_hr,
        "std_hr": std_hr,
    }
    f1_new = []
    for value in time_domain_features.values():
        array = np.array(value)
        f1_new.append(array)
    arr = np.array((f1_new))
    arr = np.nan_to_num(arr)
    return arr


def extract_frequency_domain_features(ecg_signal, fs):
    """
    Extract frequency domain features from ECG signal.
    :param ecg_signal: Input ECG signal.
    :param fs: Sampling frequency.
    :return: List of frequency domain statistical features.
    """
    f, Pxx_den = welch(ecg_signal, fs, nperseg=1024)

    mean_psd = np.mean(Pxx_den)
    std_psd = np.std(Pxx_den)
    skew_psd = skew(Pxx_den)
    kurt_psd = kurtosis(Pxx_den)

    # ------ additional features  ---------
    N = len(ecg_signal)
    # Apply FFT to the signal
    fft_signal = np.fft.fft(ecg_signal)
    freqs = np.fft.fftfreq(N, 1 / fs)
    # Only take the positive half of frequencies and corresponding FFT values
    positive_freqs = freqs[:N // 2]
    positive_fft_signal = np.abs(fft_signal[:N // 2])
    # Power Spectral Density (PSD) calculation
    psd = np.abs(positive_fft_signal) ** 2 / N  # Adjust PSD calculation
    # Band Power in the range 0.5 Hz to 50 Hz (for instance)
    mask = (positive_freqs >= 0.5) & (positive_freqs <= 50)  # Create a mask for desired range
    if len(mask) != len(psd):
        print(f"Mask shape {len(mask)} does not match PSD shape {len(psd)}")

    # Ensure dimensions match when applying boolean indexing
    band_power = np.sum(psd[mask])
    # Extract additional features
    max_freq = positive_freqs[np.argmax(positive_fft_signal)]  # Dominant frequency
    total_power = np.sum(psd)  # Total signal power

    frequency_domain_features = [mean_psd, std_psd, skew_psd, kurt_psd, max_freq, total_power, band_power]

    return frequency_domain_features


def spectral_features(y, sr=12):  # Default sample rate is 22050 Hz
    y = np.array(y.astype(float))
    # Compute Short-Time Fourier Transform (STFT)
    try:
        D = np.abs(librosa.stft(y))
    except:
        D = y
    # Compute spectral features
    spectral_centroid = librosa.feature.spectral_centroid(S=D, sr=sr).flatten()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=D, sr=sr).flatten()
    spectral_rolloff = librosa.feature.spectral_rolloff(S=D, sr=sr, roll_percent=0.14).flatten()
    spectral_flatness = librosa.feature.spectral_flatness(S=D).flatten()
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y).flatten()

    # Return features as a dictionary
    features = np.hstack(
        [spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_flatness, zero_crossing_rate])
    features = np.nan_to_num(features)
    return features


def additional_features_HHT_DFA(x):
    # ------------------ hilbert huang -----------------------------
    fs = 100
    imfs, imfs_env, imfs_freq = hht.hilbert_huang(x, fs, num_imf=10)
    # for visualization purpose
    # visualization.plot_IMFs(x, imfs, fs, save_fig="emd.png")
    # dimensionality reduction
    pca = PCA(n_components=8)
    reduced_imfs = pca.fit_transform(imfs)
    reduced_feat = reduced_imfs.flatten()

    # ------- DFA -- Detrended Fluctuation Analysis  -------------------

    pydfa = fathon.DFA(x)
    winSizes = fu.linRangeByStep(10, 100)
    revSeg = True
    polOrd = 3
    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
    return np.hstack((reduced_feat, F))


class PPGFeatureExtractor:
    def __init__(self, fs):
        self.fs = fs

    def HRV_Features_old(self, ppg_signal):
        fs = self.fs
        # Find the peaks in the PPG signal
        peaks, _ = find_peaks(ppg_signal, distance=fs * 0.5)  # Assuming a minimum of 0.5s between peaks

        # Calculate Interbeat Intervals (IBIs) in seconds
        ibi = np.diff(peaks) / fs

        # Calculate HRV features
        sdnn = np.std(ibi)  # Standard deviation of NN intervals (SDNN)
        rmssd = np.sqrt(np.mean(np.diff(ibi) ** 2))  # Root mean square of successive differences (RMSSD)
        nn50 = np.sum(np.abs(np.diff(ibi)) > 0.05)  # Number of pairs with successive differences > 50 ms (NN50)
        pnn50 = 100 * nn50 / len(ibi)  # Proportion of NN50 (pNN50)

        # Return the HRV features as a dictionary or array
        hrv_features = np.array([sdnn, rmssd, nn50, pnn50])
        return hrv_features

    def extract_temporal_features(self, ppg_signal):
        fs = self.fs
        # Duration of the signal
        duration = len(ppg_signal) / fs

        # Mean of the signal
        mean_val = np.mean(ppg_signal)

        # Standard deviation of the signal
        std_val = np.std(ppg_signal)

        # Skewness
        skewness = np.mean(((ppg_signal - mean_val) / std_val) ** 3)

        # Kurtosis
        kurtosis = np.mean(((ppg_signal - mean_val) / std_val) ** 4) - 3

        # Root Mean Square (RMS)
        rms_val = np.sqrt(np.mean(ppg_signal ** 2))

        # Mean of absolute values
        mean_abs = np.mean(np.abs(ppg_signal))

        # Zero Crossing Rate
        zero_crossings = np.sum(np.diff(np.sign(ppg_signal)) != 0) / duration

        # Return the temporal features as an array
        temporal_features = np.array([mean_val, std_val, skewness, kurtosis, rms_val, mean_abs, zero_crossings])

        return temporal_features

    # Spectral feature extraction for PPG signal using Welch's method
    def extract_spectral_features(self, ppg_signal):
        fs = self.fs
        # Compute the power spectral density using Welch's method
        freqs, psd = welch(ppg_signal, fs, nperseg=1024)

        # Define frequency bands
        lf_band = (0.04, 0.15)  # Low-frequency band (0.04 - 0.15 Hz)
        hf_band = (0.15, 0.4)  # High-frequency band (0.15 - 0.4 Hz)

        # Find indices corresponding to the LF and HF bands
        lf_idx = np.logical_and(freqs >= lf_band[0], freqs <= lf_band[1])
        hf_idx = np.logical_and(freqs >= hf_band[0], freqs <= hf_band[1])

        # Calculate power in the LF and HF bands
        lf_power = np.trapz(psd[lf_idx], freqs[lf_idx])
        hf_power = np.trapz(psd[hf_idx], freqs[hf_idx])

        # Calculate the LF/HF ratio
        lf_hf_ratio = lf_power / hf_power if hf_power != 0 else 0

        # Return the spectral features as a dictionary or array
        spectral_features = np.array([lf_power, hf_power, lf_hf_ratio])
        return spectral_features

    # Function to calculate Beats Per Minute (BPM) from PPG signal
    def calculate_bpm(self, ppg_signal):
        fs = self.fs
        # Detect peaks in the PPG signal (these correspond to heartbeats)
        peaks, _ = find_peaks(ppg_signal, distance=fs * 0.5)  # Assuming a minimum of 0.5 seconds between peaks

        # Calculate the interbeat intervals (IBIs) in seconds
        if len(peaks) < 2:
            # If fewer than 2 peaks are detected, BPM cannot be computed accurately
            return 0

        ibi = np.diff(peaks) / fs  # IBI in seconds (intervals between consecutive peaks)

        # Calculate the average IBI
        avg_ibi = np.mean(ibi)  # Average interval between beats

        # Calculate BPM using the formula: BPM = 60 / IBI
        bpm = 60 / avg_ibi

        return bpm

    # Dummy BPM value

    def crest_to_crest_interval_features(self, ppg_signal):
        fs = self.fs

        # Detect peaks (crests) in the PPG signal
        peaks, _ = find_peaks(ppg_signal, distance=fs * 0.5)  # Assuming a minimum 0.5s between peaks

        # Calculate the time intervals between consecutive peaks (crest-to-crest intervals)
        if len(peaks) < 2:
            # If fewer than 2 peaks are detected, return empty features
            return [0, 0, 0, 0]

        crest_intervals = np.diff(peaks) / fs  # Convert intervals to seconds

        # Extract statistical features from the crest-to-crest intervals
        mean_interval = np.mean(crest_intervals)
        std_interval = np.std(crest_intervals)
        min_interval = np.min(crest_intervals)
        max_interval = np.max(crest_intervals)

        # Return the features as a list
        features = [mean_interval, std_interval, min_interval, max_interval]

        return features

    def statistical_features_1lead(self, signal):
        return np.array([
            np.mean(signal),
            np.std(signal),
            np.var(signal),
            skew(signal),
            kurtosis(signal),
            np.median(signal),
            np.min(signal),
            np.max(signal),
            np.max(signal) - np.min(signal),
            np.sqrt(np.mean(np.square(np.array(signal))))
        ])

    def Stastical_features(self, final_signal):
        features_all_leads = []

        lead_signal = final_signal
        feats = self.statistical_features_1lead(lead_signal)
        features_all_leads.append(feats)

        features_all_leads = np.array(features_all_leads)

        feat1 = np.average(features_all_leads, axis=0)

        return feat1

    def VGG16_Features(self, signal):
        # Reshape the signal to match VGG16 input dimensions (224x224x3)

        signal = np.asarray(signal, dtype=np.float32)
        wav = Wavelet('gmw')

        Tx, scales = cwt(signal, wavelet=wav, scales='log', nv=32)

        TF = np.abs(Tx)
        TF = np.log1p(TF)
        TF = (TF - TF.min()) / (TF.max() - TF.min())

        TF_224 = cv2.resize(TF, (224, 224), interpolation=cv2.INTER_CUBIC)
        TF_rgb = np.repeat(TF_224[:, :, None], 3, axis=2)

        TF_rgb = TF_rgb * 255.0
        TF_rgb = np.expand_dims(TF_rgb, axis=0)
        TF_rgb = preprocess_input(TF_rgb)
        # Load VGG16 model with pre-trained ImageNet weights, excluding the fully connected layers (top)
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Define the model to return the output of the VGG16 layers
        vgg16_model = Model(inputs=base_model.input, outputs=base_model.output)
        # Get VGG16 features
        vgg16_features = vgg16_model.predict(TF_rgb)[:, :, :, :4]

        # Flatten the features to create a 1D feature vector
        vgg16_features_flattened = vgg16_features.flatten()

        return vgg16_features_flattened

    def hybrid_wavelet_hilbert_features(self, signal, wavelet1='db4', wavelet2='haar', level=4):

        features = []

        # 1) Decompose with first wavelet
        coeffs1 = pywt.wavedec(signal, wavelet1, level=level)
        for subband in coeffs1:
            analytic_sub = hilbert(subband)
            amp = np.abs(analytic_sub)
            phase = np.unwrap(np.angle(analytic_sub))

            features.append(np.mean(amp))
            features.append(np.std(amp))
            features.append(np.max(amp))
            features.append(np.min(amp))

            features.append(np.mean(phase))
            features.append(np.std(phase))

        # 2) Decompose with second wavelet
        coeffs2 = pywt.wavedec(signal, wavelet2, level=level)
        for subband in coeffs2:
            analytic_sub = hilbert(subband)
            amp = np.abs(analytic_sub)
            phase = np.unwrap(np.angle(analytic_sub))

            features.append(np.mean(amp))
            features.append(np.std(amp))
            features.append(np.max(amp))
            features.append(np.min(amp))

            features.append(np.mean(phase))
            features.append(np.std(phase))

        return np.array(features)

    def frequency_domain_features(self, signal_data):  # fs is the sampling frequency
        fs = self.fs
        N = len(signal_data)

        # Apply FFT to the signal
        fft_signal = np.fft.fft(signal_data)
        freqs = np.fft.fftfreq(N, 1 / fs)

        # Only take the positive half of frequencies and corresponding FFT values
        positive_freqs = freqs[:N // 2]  # This will have shape (N//2,)
        positive_fft_signal = np.abs(fft_signal[:N // 2])  # Same shape (N//2,)

        # Power Spectral Density (PSD) calculation
        psd = np.abs(positive_fft_signal) ** 2 / N  # Adjust PSD calculation

        # Band Power in the range 0.5 Hz to 50 Hz (for instance)
        mask = (positive_freqs >= 0.5) & (positive_freqs <= 50)  # Create a mask for desired range

        if len(mask) != len(psd):
            print(f"Mask shape {len(mask)} does not match PSD shape {len(psd)}")

        # Ensure dimensions match when applying boolean indexing
        band_power = np.sum(psd[mask])  # This will now work if both are of shape (N//2,)

        # Extract additional features
        max_freq = positive_freqs[np.argmax(positive_fft_signal)]  # Dominant frequency
        total_power = np.sum(psd)  # Total signal power

        features = [max_freq, total_power, band_power]

        return features  # , positive_freqs, positive_fft_signal


class SignalProcessor:
    def __init__(self, lowcut=0.5, highcut=300.0, fs=1000.0, order=6, data_dir="data2"):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        self.data_dir = data_dir

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def plot_signals(self, s, filtered_signal, filename):
        # Plot and save original signals
        plt.figure(figsize=(15, 10))
        plt.suptitle('Original Signal', fontsize=16)
        for i in range(s.shape[0]):
            plt.subplot(s.shape[0], 1, i + 1)
            time = np.linspace(0, s.shape[1] / self.fs, s.shape[1])
            plt.plot(time, s[i, :], label=f'Original Signal - Lead {i + 1}')
            plt.title(f'Lead {i + 1}', loc='left')  # Add lead label
            plt.xlabel('Time (seconds)')
            plt.ylabel('Signal Amplitude')
        plt.subplot(s.shape[0], 1, 1)
        plt.legend()
        original_signal_path = os.path.join("PPG signals1", f"{os.path.basename(filename).split('.')[0]}_original.png")
        plt.savefig(original_signal_path)
        plt.clf()

        # Plot and save filtered signals
        plt.figure(figsize=(15, 10))
        plt.suptitle('Butterworth Bandpass Filtered Signals', fontsize=16)
        for i in range(filtered_signal.shape[0]):
            plt.subplot(filtered_signal.shape[0], 1, i + 1)
            plt.plot(time, filtered_signal[i, :], color='orange', label=f'Filtered Signal - Lead {i + 1}')
            plt.title(f'Lead {i + 1}', loc='left')  # Add lead label
            plt.xlabel('Time (seconds)')
            plt.ylabel('Signal Amplitude')
        filtered_signal_path = os.path.join("PPG Bandpass Butterworth1",
                                            f"{os.path.basename(filename).split('.')[0]}_Butterworth_Bandpass_filtered.png")
        plt.savefig(filtered_signal_path)
        plt.clf()

    def find_pqrst_peaks(self, filtered_signal):
        r_peaks, _ = find_peaks(filtered_signal, distance=50, height=np.max(filtered_signal) * 0.5)
        if len(r_peaks) > 0:
            r_peak = r_peaks[0]
            p_peaks, _ = find_peaks(filtered_signal[:r_peak], distance=50,
                                    height=np.mean(filtered_signal) + np.std(filtered_signal))
            s_peaks, _ = find_peaks(-filtered_signal[r_peak:], distance=50,
                                    height=-np.mean(filtered_signal) - np.std(filtered_signal))
            q_peaks, _ = find_peaks(-filtered_signal[:r_peak], distance=50,
                                    height=-np.mean(filtered_signal) - np.std(filtered_signal))
            t_peaks, _ = find_peaks(filtered_signal[r_peak:], distance=50,
                                    height=np.mean(filtered_signal) + np.std(filtered_signal))
        else:
            p_peaks, q_peaks, s_peaks, t_peaks = [], [], [], []
        return p_peaks, q_peaks, r_peaks, s_peaks, t_peaks

    def determine_label(self, r_peaks, p_peaks, t_peaks, filtered_signal):
        # No heart activity
        if len(r_peaks) == 0:
            return 1, "Arrhythmia"

        # Bradycardia
        elif len(p_peaks) == 0 and np.mean(filtered_signal) < 40:
            return 1, "Arrhythmia"

        # Tachycardia
        elif len(r_peaks) > 0 and len(t_peaks) == 0 and np.mean(filtered_signal) > 140:
            return 1, "Arrhythmia"

        # Ventricular Tachycardia
        elif len(r_peaks) > 5 and np.mean(filtered_signal) > 100:
            return 1, "Arrhythmia"

        # Otherwise normal
        else:
            return 0, "Normal"

    def plot_signals_with_label(self, s, filtered_signal, label, filename):
        # Plot and save original signals with the label
        plt.figure(figsize=(15, 10))
        plt.suptitle(f'Predicted - Label: {label}', fontsize=16)
        for i in range(s.shape[0]):
            plt.subplot(s.shape[0], 1, i + 1)
            time = np.linspace(0, s.shape[1] / self.fs, s.shape[1])
            plt.plot(time, s[i, :], label=f'Original Signal - Lead {i + 1}')
            plt.title(f'Lead {i + 1}', loc='left')  # Add lead label
            plt.xlabel('Time (seconds)')
            plt.ylabel('Signal Amplitude')
        plt.subplot(s.shape[0], 1, 1)
        plt.legend()
        plt.tight_layout()

        output_dir = "Original_Signals_With_Labels"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        labeled_signal_path = os.path.join(output_dir,
                                           f"{os.path.basename(filename).split('.')[0]}_Predicted_Labels_{str(label).replace('/', '_')}.png")
        plt.savefig(labeled_signal_path)
        plt.clf()

    def plot_pqrst_peaks(self, time, filtered_signal, p_peaks, q_peaks, r_peaks, s_peaks, t_peaks, label, filename,
                         leads, p=0):
        if p:
            plt.figure(figsize=(15, 10))
            plt.suptitle(f'Filtered Signal with PQRS Peaks - {label}', fontsize=16)

            for i, lead in enumerate(leads):
                plt.subplot(len(leads), 1, i + 1)
                plt.plot(time, filtered_signal[lead], label=f'Filtered Signal Lead {lead + 1}', alpha=0.5)
                plt.plot(time[p_peaks], filtered_signal[lead][p_peaks], 'go', label='P Peaks')
                plt.plot(time[q_peaks], filtered_signal[lead][q_peaks], 'mo', label='Q Peaks')
                plt.plot(time[r_peaks], filtered_signal[lead][r_peaks], 'ro', label='R Peaks')
                plt.plot(time[s_peaks], filtered_signal[lead][s_peaks], 'co', label='S Peaks')
                plt.plot(time[t_peaks], filtered_signal[lead][t_peaks], 'bo', label='T Peaks')
                plt.title(f'Lead {lead + 1}')
                plt.title(f' - Lead {lead}', loc='left')
                # Add lead label
                plt.xlabel('Time (seconds)')
                plt.ylabel('Signal Amplitude')
                plt.legend()
                plt.grid(True)
            plt.tight_layout()

            pqrst_path = os.path.join("PPG PQRST Segmented", f"{os.path.basename(filename).split('.')[0]}_pqrst.png")
            plt.savefig(pqrst_path)
            plt.close()

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    def preprocess_signals(self, db, s=1):
        # Initialize lists to store features and labels
        Features = []
        Labels = []
        annot = []
        if s:
            if db == 'Physionet':
                directory = glob("Dataset/training/training/*.mat")[:20]
                feature_extractor = PPGFeatureExtractor(self.fs)

                final_hrv_features = []
                final_haar_wavelet_feat = []
                final_time_domain_feat = []
                final_statistical_feat = []
                final_deep_vggish_feat = []

                complete_signal_paths = []
                for filename in directory:
                    # Load the .mat file
                    mat_data = scipy.io.loadmat(filename)
                    s = mat_data['val']
                    s = s[:, :10000]  # Ensure the signal length is 5000
                    full_features = []
                    full_label = []
                    # Apply the bandpass filter to all leads
                    filtered_signal = np.zeros_like(s)
                    for i in range(s.shape[0]):
                        filtered_signal[i, :] = self.butter_bandpass_filter(s[i, :], self.lowcut, self.highcut, self.fs,
                                                                            self.order)
                    # --------- plot the signal original and butter bandpass filtered
                    # self.plot_signals(s, filtered_signal, filename)
                    o1, o2, o3, o4 = [], [], [], []
                    count = 0
                    for i in range(filtered_signal.shape[0]):
                        print(filtered_signal.shape[0])
                        final_r_peak, final_q_peak, final_p_peak, final_s_peak, final_t_peak = [], [], [], [], []
                        for leads1 in range(filtered_signal.shape[0]):
                            p_peaks, q_peaks, r_peaks, s_peaks, t_peaks = find_pqrst_peaks1(filtered_signal[leads1, :])
                            final_r_peak.append(r_peaks)
                            final_p_peak.append(p_peaks)
                            final_q_peak.append(q_peaks)
                            final_s_peak.append(s_peaks)
                            final_t_peak.append(t_peaks)

                            if count == 0:
                                label, name = self.determine_label(r_peaks, p_peaks, t_peaks, filtered_signal[i])
                                Labels.append(label)
                                count = count+1

                        final_r_peak_splitted, final_q_peak_splitted, final_p_peak_splitted, final_s_peak_splitted, final_t_peak_splitted = [], [], [], [], []

                        r_peak_splitted, p_peak_splitted, s_peak_splitted, q_peak_splitted, t_peak_splitted = split_peaks(
                            filtered_signal, final_r_peak, final_p_peak, final_s_peak, final_q_peak, final_t_peak)

                        peak_splits = [r_peak_splitted, p_peak_splitted, s_peak_splitted, q_peak_splitted,
                                       t_peak_splitted]

                        temp_hrv_features = []
                        temp_haar_wavelet_feat = []
                        temp_time_domain_feat = []
                        temp_statistical_feat = []
                        temp_deep_vggish_feat = []
                        sig_ = filtered_signal[i].flatten()
                        for index11, splits in enumerate(peak_splits):
                            f4 = feature_extractor.VGG16_Features(sig_)
                            temp_deep_vggish_feat.append(f4)
                            f11 = []
                            f22 = []
                            f33 = []
                            for s in splits:
                                if len(s) > 0:
                                    f1 = HRV_Features(s)
                                    f2 = feature_extractor.Stastical_features(s)
                                    f3 = feature_extractor.hybrid_wavelet_hilbert_features(s)
                                    f11.append(f1)
                                    f22.append(f2)
                                    f33.append(f3)

                                temp_hrv_features.append(np.mean(f11, axis=0))
                                temp_haar_wavelet_feat.append(np.mean(f33, axis=0))
                                # temp_time_domain_feat.append(f3)
                                temp_statistical_feat.append(np.mean(f22, axis=0))

                        o1.append(np.mean(temp_hrv_features, axis=0))
                        o2.append(np.mean(temp_haar_wavelet_feat, axis=0))
                        # final_time_domain_feat.append(temp_time_domain_feat)
                        o3.append(np.mean(temp_statistical_feat, axis=0))
                        o4.append(np.mean(temp_deep_vggish_feat, axis=0))
                    print("Completed signal 1")
                    print("The length of labels:",len(Labels))

                    final_hrv_features.append(np.mean(o1, axis=0))
                    final_haar_wavelet_feat.append(np.mean(o2, axis=0))
                    final_statistical_feat.append(np.mean(o3, axis=0))
                    final_deep_vggish_feat.append(np.mean(o4, axis=0))

                final_concated_features = np.hstack(
                    [np.array(final_hrv_features), np.array(final_haar_wavelet_feat), np.array(final_statistical_feat),
                     np.array(final_deep_vggish_feat)])

                np.save(f"New Dataset/Physionet/n_Features.npy", np.array(final_concated_features))
                np.save(f"New Dataset/Physionet/n_Labels.npy", np.array(Labels))


            else:
                path = 'Dataset/mimic-database-1.0.0/mimic-database-1.0.0/Data2/Data'
                all_fold = glob(path + "/**")
                for F in range(len(all_fold)):
                    fold = all_fold[F]
                    files = list()
                    # all the header files from chf dataset
                    for k in os.listdir(fold):
                        if k.endswith(".dat"):
                            files.append(k)

                    for i in tqdm(range(len(files)), desc="Preprocessing"):
                        file = files[i]
                        file = file[:-4]
                        record = wfdb.rdrecord(fold + '/' + file, sampfrom=1000, sampto=6000)
                        record = wfdb.rdrecord(fold + '/' + file)
                        ann = wfdb.rdann(fold + '/' + file, 'al', sampfrom=1000, sampto=6000)
                        if not ann.aux_note:
                            lab = 0
                        else:
                            # lab = max(ann.aux_note)
                            lab = 1

                        # -------------------------------------------------
                        s = record.p_signal
                        s = s.T

                        # Apply the bandpass filter to all leads
                        filtered_signal = np.zeros_like(s)
                        for h in range(s.shape[0]):
                            filtered_signal[h, :] = self.butter_bandpass_filter(s[h, :], self.lowcut, self.highcut,
                                                                                self.fs, self.order)
                        filtered_signal = np.nan_to_num(filtered_signal)
                        # if filteres_signal == Nan:
                        #     break
                        # else:
                        feature_extractor = PPGFeatureExtractor(self.fs)
                        for w in range(filtered_signal.shape[0]):
                            annot.append(lab)
                            # print(filtered_signal.shape[0])
                            p_peaks, q_peaks, r_peaks, s_peaks, t_peaks = self.find_pqrst_peaks(filtered_signal[w])
                            self.plot_pqrst_peaks(
                                time=np.linspace(0, s.shape[1] / self.fs, s.shape[1]),
                                filtered_signal=filtered_signal,
                                p_peaks=p_peaks, q_peaks=q_peaks, r_peaks=r_peaks,
                                s_peaks=s_peaks, t_peaks=t_peaks, label=lab,
                                filename='filename', leads=range(filtered_signal.shape[0])
                            )
                            sig_ = filtered_signal[w].flatten()
                            f1 = HRV_Features(sig_)
                            f2 = extract_frequency_domain_features(sig_, self.fs)
                            f3 = spectral_features(sig_)

                            f4 = feature_extractor.calculate_bpm(sig_)
                            f5 = feature_extractor.crest_to_crest_interval_features(sig_)
                            f6 = feature_extractor.VGG16_Features(sig_)

                            f7 = additional_features_HHT_DFA(sig_)
                            cprint(f'Features extracted --- {i + 1}', 'green')
                            cprint('-------------------------------------------', 'magenta')
                            features = np.hstack([f1, f2, f3, f4, f5, f6, f7])
                            Features.append(features)
            # if db == 'Physionet':
            #     np.save(f"n_Features.npy", Features)
            #     np.save(f"n_Labels.npy", Labels)
            # else:
            #     max_length = max(len(arr) for arr in Features)
            #     all_ = [np.pad(arr, (0, max_length - len(arr)), mode='constant') for arr in Features]
            #     np.save(f"n_Features.npy", all_)
            #     np.save(f"n_Labels.npy", annot)
            #     cprint(f"Features saved to ----> n_Features.npy", 'green')
            #     cprint(f"Features saved to ----> n_Labels.npy", 'green')
