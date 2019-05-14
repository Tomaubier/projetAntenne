from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import scipy.signal as sg
import numpy.fft as fft
import projetdef as pc
import numpy as np

frontMic = 0

def index(Fk, fmax):
    ii = 0
    while Fk[ii] < fmax: ii += 1
    return ii

# Axes temporel / fréquentiel
Fs, Ntfd = 44100, 2**13
t = np.arange(Fs)/Fs
Fk = np.arange(Ntfd)*Fs/Ntfd

# Parametres antenne / source
M, R = 6, 4.32e-2
amp_s, f_s, ang_s = 1, 600, np.pi*0

# Création de l'antenne / source #
antenne = pc.CircularArray(M, R)
x, y = antenne.coordinates()
source = pc.PlaneWave(amp_s, f_s, ang_s)

print(pc.freq_aliasing(x, y))
################################################################################
# Calcul du traitement
################################################################################

# Calcul des gains dans la bande de fréquence [500, 700[
ii_min, ii_max = index(Fk, 500), index(Fk, 700)
H = np.zeros((M, Fk.size), dtype='cfloat')
for ii, freq in enumerate(Fk[ii_min:ii_max], ii_min):
    H[:, ii] = antenne.filter_order3(freq, frontMic)

pt_interp = 50 # Nb de points de l'interpolation
del_bef, del_aft = np.arange(ii_min - pt_interp, ii_min),  np.arange(ii_max, ii_max + pt_interp)
del_elt = np.stack([del_bef, del_aft]) # Elements à supprimer pour l'interpolation

# Interpolation
Fk_inter = np.delete(Fk, del_elt)
for mic in range(M):
    H_inter = np.delete(H[mic], del_elt)
    f = interp1d(Fk_inter, H_inter, kind='cubic')
    H[mic, :int(Ntfd/2)-1] = f(Fk)[:int(Ntfd/2)-1]

# Symétrie Hermitienne
for ii, freq in enumerate(Fk[1:int(Fk.size/2)], 1):
    H[:, -ii] = np.conj(H[:, ii])

# Réponse impulsionnelle h
h = np.real(fft.ifft(np.conj(H)))

# Vérification de la partie imaginaire nulle, DC et Nyquist (en enlevant np.real())
# print(np.round(np.imag(h[:, ii_min-pt_interp:ii_max+pt_interp]), 16) == 0)
# print(h[:, 0], h[:, int(Ntfd/2)] == 0)

# IR non causale vers causale (equivalent a circshift sous Matlab)
h = np.roll(h, int(h.size/2), axis=1)
hTrimmed = h[:, 3700:-3700]

################################################################################
# %% Application du traitement avec les RI et fftconvolve
################################################################################

# Création des signaux pour les M microphones
micros = np.real(source.field(x, y, t))

# Traitement de l'antenne, avec z le signal de sortie
# NB : scipy 1.2.1 ou plus est requis pour le support de l'argument axes
z = np.sum(sg.fftconvolve(micros, h, mode='same', axes=1), axis=0)
zHTrimmed = np.sum(sg.fftconvolve(micros, hTrimmed, mode='same', axes=1), axis=0)

################################################################################
# Figures
################################################################################


# %% Champs de Pression

x, y = np.linspace(-2, 2, 200), np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y)


pWave = pc.PlaneWave(1, 10, np.pi/3)
pWave.field(X, Y, np.array([0]))
#
# %% Motif de faisceau

theta = np.linspace(0, 2*np.pi, 500)
beamA = antenne.pattern(theta, index(Fk, f_s), frontMic)
plt.figure(figsize=(9, 9))
ax = plt.subplot(111, projection='polar')
plt.plot(theta, 20*np.log10(beamA), 'r', label=r'$\mathcal{B}_1(\theta - \phi_1)$')
ax.set_rmin(-20*np.log10(2))
ax.set_rticks([0, -10, -20, -30, -40, -50])
ax.set_yticklabels([r'0 dB', r'-10', r'-20', r'-30', r'-40', r'-50'])
ax.set_theta_zero_location("E")
ax.set_rlabel_position(15)
ax.set_title('{} Microphones / Motif de faisceau'.format(M))
plt.thetagrids(np.arange(0, 360, 30))
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.tight_layout()
plt.show()

# %% Réponses impulsionnelles

fig, ax = plt.subplots(nrows=1, figsize=(11, 6))
for each in range(M):
    ax.plot(np.arange(h.shape[1]), h[each], label=r'$h_{}[n]$'.format(each+1))
plt.legend()
ax.set_xlabel(r'$n$')
ax.set_ylabel(r'Amplitude')
ax.set_xlim([0, h.shape[1]])
ax.set_title('{} Microphones / Reponses impulsionnelles'.format(M))
plt.grid()
plt.tight_layout()
plt.show()

# %% Représentation du signal de sortie en fonction de l'angle de la source

fig, ax = plt.subplots(nrows=1, figsize=(11, 6))
ax.plot(t, micros[0], label=r'$M_{1}$')
ax.plot(np.arange(z.size)/Fs, zHTrimmed, 'k', label=r'$z[n]$')
ax.plot(np.arange(z.size)/Fs, zHTrimmed - z)
ax.plot(np.arange(z.size)/Fs, z, 'r--', label=r'$z[n]$')
plt.legend(loc=2)
ax.set_xlim([0.09, 0.10])
ax.set_xlabel(r'Temps en $s$')
ax.set_ylabel(r'Amplitude')
plt.grid()
ax.set_title('{} Microphones / Angle Source : {:.3} rad'.format(M, ang_s))
plt.tight_layout()
plt.show()

# %%

hTrimmed.shape[1]*2

plt.figure(figsize=(13, 7))
plt.plot(fft.fft(h[0]), '*')
plt.plot(fft.fft(hTrimmed[0]), '*')
plt.show()
