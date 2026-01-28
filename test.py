import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from ssqueezepy import cwt, Wavelet
import cv2
#
# ---------------------------
# 1. Load ECG
# ---------------------------
mat_data = scipy.io.loadmat("Dataset/training/training/a105l.mat")
x = mat_data['val'][0]        # Lead-1
x = x[:10000]

fs = 360   # MIT-BIH ECG sampling rate

# ---------------------------
# 2. Morse (GMW) wavelet
# ---------------------------
wav = Wavelet('gmw')

# ---------------------------
# 3. Continuous Wavelet Transform
# ---------------------------
Tx, scales = cwt(
    x,
    wavelet=wav,
    scales='log',
    nv=32
)

# ---------------------------
# 4. True time-frequency map
# ---------------------------
TF = np.abs(Tx)   # keep time–frequency localization

# ---------------------------
# 5. Log compression
# ---------------------------
TF = np.log1p(TF)

# ---------------------------
# 6. Normalize
# ---------------------------
TF = (TF - TF.min()) / (TF.max() - TF.min())

# ---------------------------
# 7. Resize for CNN
# ---------------------------
TF_224 = cv2.resize(TF, (224,224), interpolation=cv2.INTER_CUBIC)
TF_rgb = np.repeat(TF_224[:,:,None], 3, axis=2)

print("CNN image shape:", TF_rgb.shape)

# ---------------------------
# 8. Display (ssqueezepy-style)
# ---------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(x)
plt.title("ECG Signal (Lead-1)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.subplot(1,2,2)
plt.imshow(
    TF,
    aspect='auto',
    origin='lower',
    cmap='turbo'
)
plt.title("Morse (GMW) Time-Frequency Feature Map")
plt.xlabel("Time")
plt.ylabel("Scale (pseudo-frequency)")
plt.colorbar(label="Magnitude")

plt.tight_layout()
plt.show()
#
#
# # https://github.com/OverLordGoldDragon/ssqueezepy


import numpy as np
import pywt
from scipy.signal import hilbert

def hybrid_wavelet_hilbert_features(signal, wavelet1='db4', wavelet2='haar', level=4):

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

mat_data = scipy.io.loadmat("Dataset/training/training/a105l.mat")
x = mat_data['val'][0]        # Lead-1
x = x[:10000]

fs = 360

out_feat = hybrid_wavelet_hilbert_features(x)
print(np.array(out_feat).shape)