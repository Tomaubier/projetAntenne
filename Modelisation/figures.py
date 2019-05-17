import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) # Set project path
import differentialarray as da
import matplotlib.pyplot as plt
import scipy.signal as sg
import numpy as np

# Constantes
n_interp = 100
steering_mic, Fs, Ntfd, freqBand, interp = 1, 44100, 44100, (441, 1851), (True, n_interp)

# Axes temporel / angulaire #
t = np.arange(Fs)/Fs
n = np.arange(Fs)
theta = np.linspace(0, 2*np.pi, 500)
Fk = np.arange(Ntfd)*Fs/Ntfd


################################################################################
#           FIGURES : CHAMP PRESSION
################################################################################

#------------------------------------------------------------------------------#
# %%  Figure 1 : Champ de pression
#------------------------------------------------------------------------------#

# fig, ax0 = plt.subplots(nrows=1)
# fig = plt.gcf()
# fig.canvas.set_window_title('Champ de pression')
# antenne = da.DifferentialArray(steering_mic, Fs, Ntfd, freqBand, interp)
# x, y = antenne.coordinates()
# x_grid = np.linspace(-antenne.R*(1 + 0.1), antenne.R*(1 + 0.1), 500)
# x_grid, y_grid = np.meshgrid(x_grid, x_grid)
# ax0.set_xticks([-antenne.R, 0, antenne.R])
# ax0.set_xticklabels([r'$-R$', '0', r'$R$'])
# ax0.set_yticks([-antenne.R, 0, antenne.R])
# ax0.set_yticklabels([r'$-R$', '0', r'$R$'])
# ax0.set_xlim([-antenne.R*(1 + 0.1), antenne.R*(1 + 0.1)])
# ax0.set_ylim([-antenne.R*(1 + 0.1), antenne.R*(1 + 0.1)])
# ax0.set_xlabel(r'Position suivant $x$')
# ax0.set_ylabel(r'Position suivant $y$')
# S = da.PlaneWave(1, 500, np.pi/2)
# print(2*np.pi*S.f*antenne.R/342)
# field = np.real(S.static_field(x_grid, y_grid))
# c = ax0.pcolormesh(x_grid, y_grid, field, cmap='coolwarm')
# cbar = fig.colorbar(c, ticks=[np.min(field), 0, np.max(field)])
# cbar.ax.set_yticklabels([r'$P_{min}$', r'$0$', r'$P_{max}$'])
# for ii in range(antenne.M):
#     ax0.plot(x[ii], y[ii], 'oC{}'.format(ii), label=r'$M_{}$'.format(ii+1))
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join('Figures', '1-1-ChampPression.png'), facecolor='w', edgecolor='w')
# plt.show()
#
# #------------------------------------------------------------------------------#
# # %%  Figure 2 : Pression des microphones
# #------------------------------------------------------------------------------#
# fig, ax1 = plt.subplots(nrows=1)
# fig = plt.gcf()
# fig.canvas.set_window_title('Pression des microphones')
# antenne = da.DifferentialArray()
# x, y = antenne.coordinates()
# source = da.PlaneWave(ang=np.pi/2)
# micros = np.real(source.field(x, y, t))
# for ii in range(antenne.M):
#     ax1.plot(t*Fs, micros[ii, :], '*-', label=r'$y_{}[n]$'.format(ii+1))
# ax1.set_xlim([0, (1/source.f)*Fs])#])
# ax1.set_ylim([min(micros[0])*(1+0.05),  max(micros[0])*(1+0.05)])
# ax1.set_yticks([min(micros[0]), 0, max(micros[0])])
# ax1.set_yticklabels([r'$P_{min}$', r'$0$', r'$P_{max}$'])
# ax1.set_xlabel(r"Échantillons $n$"), ax1.set_ylabel('Pression en Pa')
# ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# ax1.grid()
# plt.tight_layout()
# plt.savefig(os.path.join('Figures', '1-2-PressionMicros.eps'), facecolor='w', edgecolor='w')
# plt.show()
#
#
# ################################################################################
# #           FIGURES : TRAITEMENT UNE FREQUENCE
# ################################################################################
#
# #------------------------------------------------------------------------------#
# # %%  Figure 1 : Exemple de motif de faisceau
# #------------------------------------------------------------------------------#
#
# fig = plt.figure()
# fig = plt.gcf()
# fig.canvas.set_window_title('Exemple de motif de faisceau d'ordre 3')
# ax = plt.subplot(111, projection='polar')
# plt.plot(theta, 20*np.log10(beamA), 'r', label=r'$\mathcal{B}_3(\theta - \phi_1)$')
# ax.set_rmin(-20*np.log10(2))
# ax.axvline()
# ax.set_rticks([0, -10, -20, -30, -40, -50])
# ax.set_yticklabels([r'0 dB', r'-10', r'-20', r'-30', r'-40', r'-50'])
# ax.set_theta_zero_location("E")
# ax.set_rlabel_position(15)
# ax.annotate('Lobe Principal',
#             xy=(np.deg2rad(-10), 0),  # theta, radius
#             xytext=(1, 0.4),    # fraction, fraction
#             textcoords='figure fraction',
#             arrowprops=dict(facecolor='black', shrink=1),
#             horizontalalignment='right',
#             verticalalignment='bottom',
#             )
# ax.annotate('Zéro',
#             xy=(np.deg2rad(90), 0),  # theta, radius
#             xytext=(0.4, 0.4),    # fraction, fraction
#             textcoords='figure fraction',
#             arrowprops=dict(facecolor='black', shrink=1),
#             horizontalalignment='left',
#             verticalalignment='top',
#             )
# plt.thetagrids(np.arange(0, 360, 30))
# plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
# plt.tight_layout()
# plt.savefig(os.path.join('Figures', 'beam.eps'), facecolor='w', edgecolor='w')
# plt.show()
#
#
# #------------------------------------------------------------------------------#
# # %%  Figure 2 : Modélisation de l'expérience / motif de faisceau
# #------------------------------------------------------------------------------#
#
# fig = plt.gcf()
# fig.canvas.set_window_title("Modélisation de l'expérience / motif de faisceau")
# angles = np.linspace(0, 2*np.pi, 72)
#
# f_s = 500
# a = da.DifferentialArray(steering_mic, Fs, Ntfd, freqBand, interp)
# beamA = a.beampattern(theta, 500)
# beamB = a.beampattern(theta, 1851)
# h = a.impulse_responses()
# x, y = a.coordinates()
#
# ax = plt.subplot(111, projection='polar')
# ax.plot(theta,20*np.log10(beamA), 'r', label=r'$B_1[k, \theta_s = \phi_{}]$'.format(steering_mic))
# #plt.plot(0, 1, 'or', label=r'Expérience')
# #for ii, each in enumerate(angles):
#     #S = da.PlaneWave(1, f_s, each)
#     #m = np.real(S.field(x, y, t))
#     #micros = np.real(S.field(x, y, t))
#     #z = np.sum(sg.fftconvolve(micros, h, mode='same', axes=1), axis=0)
#     #ax.plot(each, max(z[3*f_s:10*f_s]), 'or')
# ax.set_rmin(-50)
# ax.set_rticks([0, -10, -20, -30, -40, -50])
# ax.set_yticklabels([r'0 dB', r'-10', r'-20', r'-30', r'-40', r'-50'])
# ax.set_theta_zero_location("E")
# ax.set_rlabel_position(15)
# plt.thetagrids(np.arange(0, 360, 30))
# plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
# plt.tight_layout()
# plt.savefig(os.path.join('Figures', '2-2-ModelisationExp.svg'), facecolor='w', edgecolor='w')
# plt.show()
#
# ################################################################################
# #           FIGURES : TRAITEMENT LARGE BANDE
# ################################################################################
#
# #------------------------------------------------------------------------------#
# # %%  Figure 1 : Module des filtres à appliquer à chaque microphone
# #------------------------------------------------------------------------------#
# a = da.DifferentialArray(steering_mic, Fs, Ntfd, freqBand=(1, Fk.size/2), interp=(False, 10))
# x, y = a.coordinates()
# delta = np.sqrt((x[1]-x[0])**2 + (y[1]-y[0])**2)
# frep = 342/(2*delta)
# matriceH = a.filter_matrix()
# print(matriceH.shape, frep)
# fig, ax = plt.subplots(nrows=1)
# fig = plt.gcf()
# fig.canvas.set_window_title('Module des filtres à appliquer à chaque microphone')
# for mic in range(a.M):
#     ax.plot(Fk, 20*np.log10(np.abs(matriceH[mic, :])), '*-', label=r'$|H_{}[F_k]|$'.format(mic+1))
# ax.set_xticklabels([r'$0$', r'${k_i}$ $\Delta f$', r'${k_f}$ $\Delta f$', r'$f_{rep}$', r'$F_s/2$'])
# ax.set_xticks([0, 441, 1851, frep, Fs/2])
# ax.set_ylabel(r'Module en $dB$')
# ax.set_xlabel(r'Fréquence $F_k$ en $Hz$')
# ax.set_xlim([0, Fs/2])
# plt.legend()
# ax.grid()
# ax.text((1851-441)/2 -100, 25, 'Bande de fréquence estimée', color='k')
# plt.axvline(frep, ls='--', color='r')
# plt.axvline(441, color='k')
# plt.axvline(1851, color='k')
# ax.axvspan(441, 1851, alpha=0.3, color='orange')
#
# plt.gca().get_xticklabels()[1].set_color('k')
# plt.gca().get_xticklabels()[2].set_color('k')
# plt.gca().get_xticklabels()[3].set_color('red')
# plt.tight_layout()
# plt.savefig(os.path.join('Figures', '3-1-ModuleFiltres.svg'), facecolor='w', edgecolor='w')
# plt.show()
#
# #------------------------------------------------------------------------------#
# # %%  Figure 2 : interpolation
# #------------------------------------------------------------------------------#
# a = da.DifferentialArray(steering_mic, Fs, Ntfd, freqBand, interp=(False, n_interp))
# b = da.DifferentialArray(steering_mic, Fs, Ntfd, freqBand, interp=(True, n_interp))
# matriceH = a.filter_matrix()
# matriceH2 = b.filter_matrix()
#
# fig, (ax, ax2) = plt.subplots(ncols=2)
# fig = plt.gcf()
# fig.canvas.set_window_title("Représentation de l'interpolation")
# ax.plot(Fk, np.abs(matriceH[0, :]), '*-', label=r'Sans interpolation')
# ax.plot(Fk, np.abs(matriceH2[0, :]), '*-', label=r'Avec interpolation')
# ax.set_xlim([(a.ii_min - n_interp)*(1-0.01)*Fs/Ntfd, (a.ii_min)*Fs/Ntfd*(1+0.01)])
# ax2.plot(Fk, np.abs(matriceH[0, :]), '*-', label=r'Sans interpolation')
# ax2.plot(Fk, np.abs(matriceH2[0, :]), '*-', label=r'Avec interpolation')
# ax2.set_xlim([(a.ii_max)*(1-0.005)*Fs/Ntfd, (a.ii_max + n_interp)*(1+0.005)*Fs/Ntfd])
# ax.grid()
# ax.set_xticklabels([r'$k_i$ - $N_{interp}$', r'$k_i$'])
# ax2.set_xticklabels([r'$k_f$', r'$k_f$ + $N_{interp}$'])
# ax.set_xticks([(a.ii_min - n_interp)*Fs/Ntfd, (a.ii_min)*Fs/Ntfd])
# ax2.set_xticks([a.ii_max*Fs/Ntfd, (a.ii_max+ n_interp)*Fs/Ntfd])
# ax.set_ylabel(r'Module $|H_1[k]|$')
# ax.set_xlabel(r'Échantillons $k$')
# ax2.set_ylabel(r'Module $|H_1[k]|$')
# ax2.set_xlabel(r'Échantillons $k$')
# ax2.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join('Figures', '3-2-Interpolation.svg'), facecolor='w', edgecolor='w')
# plt.show()
#
# #------------------------------------------------------------------------------#
# # %%  Figure 3 : Réponses impulsionnelles
# #------------------------------------------------------------------------------#
# t = np.arange(Ntfd)
# a = da.DifferentialArray(steering_mic, Fs, Ntfd, freqBand, interp=(True, n_interp))
# matriceH = a.impulse_responses()
# fig, ax = plt.subplots(ncols=1)
# fig = plt.gcf()
# fig.canvas.set_window_title("Réponses impulsionnelles des microphones")
# for mic in range(a.M):
#     ax.plot(t, matriceH[mic, :], '*-', label=r'$RI_{}[n]$'.format(mic+1))
# ax.set_xlim([4000, 4200])#[(t.size//2)*(1-0.03), (t.size//2)*(1+0.03)])
# ax.grid()
# ax.set_ylabel(r'Amplitude')
# ax.set_xlabel(r'Échantillons $n$')
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join('Figures', '3-3-ImpulseResponse.svg'), facecolor='w', edgecolor='w')
# plt.show()

#------------------------------------------------------------------------------#
# %%  Figure 4 : Signal de sortie et comparaison
#------------------------------------------------------------------------------#
t = np.arange(Fs)/Fs
a = da.DifferentialArray(steering_mic=steering_mic, Fs=Fs, Ntfd=Ntfd, freqBand=freqBand,interp=(True, n_interp))
x, y = a.coordinates()

source1, source2 = da.PlaneWave(amp=1, f=500, ang=np.pi/2), da.PlaneWave(amp=1,f=510,ang=np.pi/2)
micros = np.real(source1.field(x, y, t) + source2.field(x, y, t))

output = a.dma_output(micros)
t_output = np.arange(output.size)/Fs
fig, (ax, ax1) = plt.subplots(ncols=2)
fig = plt.gcf()
fig.canvas.set_window_title("Signal de sortie")

ax.plot(t, micros[0, :], label=r'$M_{1}[n]$')
ax.plot(t_output, output,'*-', label=r'$z[n]$')

ax.set_yticklabels([r'$-P_{max}$',r'$-P_{max}/2$', 0 , r'$P_{max}/2$', r'$P_{max}$'])
ax.set_yticks([-max(micros[0, :]), -max(micros[0, :])/2, 0, max(micros[0, :])/2,max(micros[0, :])])
ax.set_xlim([0, 1])
ax.grid()
ax.set_ylabel(r'Amplitude')
ax.set_xlabel(r'Temps $n/F_{s}$ en s')

ax1.plot(t, micros[0, :], '*-', label=r'$y_{1}[n]$')
ax1.plot(t_output, output,'*-', label=r'$z[n]$')
ax1.set_yticklabels([r'$-P_{max}$',r'$-P_{max}/2$', 0 , r'$P_{max}/2$', r'$P_{max}$'])
ax1.set_yticks([-max(micros[0, :]), -max(micros[0, :])/2, 0, max(micros[0, :])/2,max(micros[0, :])])
ax1.set_xlim([0.64, 0.76])
ax1.grid()
ax1.set_ylabel(r'Amplitude')
ax1.set_xlabel(r'Temps $n/F_{s}$ en s')

plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.tight_layout()
plt.savefig(os.path.join('Figures', '3-4-OutputSignals.eps'), facecolor='w', edgecolor='w')
plt.show()
