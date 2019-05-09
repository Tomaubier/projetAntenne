import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.fftpack import fft, ifft
import differentialarrayFirstOrder as da
import os


steering_mic, Fs, Ntfd, freqBand, interp = 1, 44.1e3, 176128, [125, 1500], (True, 100)

antenne = da.DifferentialArray(steering_mic, Fs, Ntfd, freqBand, interp )
gains = antenne.filter_matrix()
Fk = np.arange(Ntfd)*Fs/Ntfd

parentPath = 'C:/Utilisateurs/Alexia Pascal/Bureau/Directivite/Measure'
MEASURE_NAME, ANGLE_STEP = 'Chirp', 5
fexp = 1000


angles = np.arange(360 // ANGLE_STEP)*ANGLE_STEP #4200
Z = np.zeros((72, Ntfd))
value41 = np.zeros(360 // ANGLE_STEP)
for ii, angle in enumerate(angles):
    name = '{}deg_{}.wav'.format(angle, MEASURE_NAME)
    _, data = wav.read(os.path.join(parentPath, name))

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

    """
    fig, ax = plt.subplots(nrows=1)
    ax.plot(Fk, np.abs(Normed_Output))
    ax.set_title("Degrés {}".format(angle))
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Frequence en Hz')
    ax.set_xlim(freqBand)
    ax.set_ylim([0, 1])
    plt.show()"""
    #wav.write(os.path.join(Path_New, name), Fs, data)

x, y = Fk, angles
X, Y = np.meshgrid(x, y)

plt.subplots(nrows=1)
plt.pcolormesh(X, Y, Z, cmap='RdBu', vmax=1)
plt.xlim(freqBand)
plt.xlabel(r'Fréquence en Hz')
plt.ylabel(r'Angle en degré')
plt.colorbar()


"""

ax.plot(np.deg2rad(angles), 20*np.log10(value41/max(value41)))
print(Fk[fexp])"""
plt.show()
