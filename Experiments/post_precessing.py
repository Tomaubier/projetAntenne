import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) # Set project path
import differentialarray.differentialarrayFirstOrder as da
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np

2**13

steering_mic, Fs, Ntfd, freqBand, interp = 1, 44.1e3, 176128, [60, 2285], (True, 100)

antenne = da.DifferentialArray(steering_mic, Fs, Ntfd, freqBand, interp )
gains = antenne.filter_matrix()
Fk = np.arange(Ntfd)*Fs/Ntfd

parentPath = os.path.abspath('./Experiments')
dataDirectoryPath = os.path.join(parentPath, 'Data')
MEASURE_NAME, ANGLE_STEP = 'ChirpNoWindow', 5
fexp = 1000


angles = np.arange(360 // ANGLE_STEP)*ANGLE_STEP #4200
Z = np.zeros((72, Ntfd))
value41 = np.zeros(360 // ANGLE_STEP)
for ii, angle in enumerate(angles):
    name = '{}deg_{}.wav'.format(angle, MEASURE_NAME)
    _, data = wav.read(os.path.join(dataDirectoryPath, MEASURE_NAME, name))

    mic_ref = data[:, 1].T
    M0 = fft(mic_ref, Ntfd)
    #MIC2 = micros[:, 6]
    #MIC3 =  micros[:, 4]
    #micros[:, 4], micros[:, 6] = MIC2, MIC3
    micros = np.delete(data, [0, 1, 3, 5, 7], axis=1).T
    MICROS = fft(micros, Ntfd)
    Output = np.sum(np.multiply(np.conj(gains), MICROS), axis=0)

    Normed_Output = Output/M0
    Z[ii, :] = np.abs(Normed_Output)
    value41[ii] = np.abs(Normed_Output[fexp])

# %% Polar plot

ax = plt.subplot(111, projection='polar') #, figsize=(12, 5))
ax.set_theta_zero_location('E')
ax.plot(np.deg2rad(angles), Z[:, np.where(Fk >= 500)[0][0]])
plt.grid(True)
plt.show()

# %%

index_freqBand = [np.where(Fk >= f)[0][0] for f in freqBand]
Zrolled = np.roll(Z[:, index_freqBand[0]:index_freqBand[1]], Z.shape[0] // 2, axis=0)

plt.subplots(nrows=1, figsize=(12, 5))
plt.imshow(Zrolled, cmap='RdBu_r', vmin=0, vmax=1, extent=[freqBand[0], freqBand[1], -175, 180], aspect=3, interpolation='nearest', origin='lower')
plt.xlabel(r'Fréquence (Hz)')
plt.ylabel(r'Angle (°)')
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(parentPath, 'Figures', '2-2-AngleFreq.eps'))
plt.show()
