import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) # Set project path
import differentialarray.differentialarrayFirstOrder as da
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np

2**13

steering_mic, Fs, Ntfd, freqBand, interp = 1, 44.1e3, 176128, [125, 1500], (True, 100)

antenne = da.DifferentialArray(steering_mic, Fs, Ntfd, freqBand, interp )
gains = antenne.filter_matrix()
Fk = np.arange(Ntfd)*Fs/Ntfd

parentPath = os.path.abspath('.')
dataDirectoryPath = os.path.join(parentPath, 'Data')
MEASURE_NAME, ANGLE_STEP = 'ChirpWindow', 5
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


    # fig, ax = plt.subplots(nrows=1)
    # ax.plot(Fk, np.abs(Normed_Output))
    # ax.set_title("Degrés {}".format(angle))
    # ax.set_ylabel('Amplitude')
    # ax.set_xlabel('Frequence en Hz')
    # ax.set_xlim(freqBand)
    # ax.set_ylim([0, 1])
    # plt.show()
    #wav.write(os.path.join(Path_New, name), Fs, data)

index_freqBand = [np.where(Fk >= f)[0][0] for f in freqBand]
Z = Z[:, index_freqBand[0]:index_freqBand[1]]

# %%
%%time
plt.subplots(nrows=1, figsize=(10, 5))
plt.pcolormesh(X, Y, Z, cmap='RdBu', vmax=1)
plt.xlim(freqBand)
plt.xlabel(r'Fréquence en Hz')
plt.ylabel(r'Angle en degré')
plt.colorbar()
plt.show()

# %%
%%time
plt.subplots(nrows=1, figsize=(12, 5))
plt.imshow(Z, cmap='RdBu', vmin=0, vmax=1, extent=[freqBand[0], freqBand[1], 0, 355], aspect=2.5, interpolation='nearest', origin='lower')
plt.xlabel(r'Fréquence en Hz')
plt.ylabel(r'Angle en degré')
plt.colorbar()
plt.savefig(os.path.join(parentPath, 'Figures', 'directivityFreq.pdf'))
plt.show()

# plt.savefig(os.path.join(parentPath, 'Figures', 'directivityFreq.pdf'))
