from scipy.interpolate import interp1d
import numpy.fft as fft
import numpy as np


def index(array, value):
    index = 0
    while array[index] < value: index += 1
    return index


class Medium:
    def __init__(self, C=342):
        super(Medium, self).__init__()
        self.C = C


class PlaneWave(Medium):
    def __init__(self, amp=1, f=500, ang=0):
        super(PlaneWave, self).__init__()
        self.amp, self.f, self.ang = amp, f, ang
        self.w = 2 * np.pi * self.f
        self.k = self.w / self.C * np.array([np.cos(self.ang), np.sin(self.ang)])

    def field(self, x, y, t):
        time = np.exp(1j*self.w*t)
        field = self.amp*np.exp(1j*np.dot(self.k, [x, y]))
        return np.dot(field.reshape(x.size, 1), time.reshape(1, t.size))


class CircularArray:
    def __init__(self, M=3, R=4.32e-2): # 1st order: M=3 / 3rd: M=6
        super(CircularArray, self).__init__()
        self.M, self.R = M, R
        self.phi = np.arange(0, 2 * np.pi, 2 * np.pi / self.M)
        self.dphi = self.phi[1] - self.phi[0]

    def coordinates(self):
        return self.R * np.cos(self.phi), self.R * np.sin(self.phi)


class DifferentialArray(CircularArray, Medium):
    def __init__(self, steering_mic=1, Fs=8000, Ntfd=2**13, freqBand=(500, 2900), interp=(True, 10)):
        super(DifferentialArray, self).__init__()
        self.Fs, self.Ntfd = Fs, Ntfd
        self.Fk = np.arange(self.Ntfd)*self.Fs / self.Ntfd
        self.tau = self.R/self.C
        self.steering_mic = steering_mic - 1
        self.interp, self.pt_interp = interp
        self.ii_min, self.ii_max = index(self.Fk, freqBand[0]), index(self.Fk, freqBand[1])

    def steering_vector(self, angle, f):
        d = [np.exp(-1j*2*np.pi*f*self.tau*np.cos(angle - phi)) for phi in self.phi]
        return np.array(d, dtype='cfloat')

    def filter_vector(self, f):
        A, b = np.zeros((self.M, self.M), dtype='cfloat'), np.zeros(self.M, dtype='cfloat')
        b[0] = 1
        # 1st order
        A[-1] = np.roll(np.array([0, 1, -1], dtype='cfloat'), self.steering_mic, None)
        # 3rd order
        # A[-2] = np.roll(np.array([0, 1, 0, 0, 0, -1], dtype='cfloat'), self.steering_mic, None)
        # A[-1] = np.roll(np.array([0, 0, 1, 0, -1, 0], dtype='cfloat'), self.steering_mic, None)

        theta_c = np.array([0, np.pi]) + self.dphi*self.steering_mic # 1st order: [0, np.pi] / 3rd: [0, np.pi/2, 2*np.pi/3, np.pi]
        for ii, angle in enumerate(theta_c):
            A[ii] = self.steering_vector(angle, f)
        h = np.linalg.solve(A, b)
        return h

    def filter_matrix(self):
        H = np.zeros((self.M, self.Fk.size), dtype='cfloat')
        for ii, freq in enumerate(self.Fk[self.ii_min:self.ii_max], self.ii_min):
            H[:, ii] = self.filter_vector(freq)
        if self.interp:
            H = self.interpolation(H)
        for ii, freq in enumerate(self.Fk[1:int(self.Fk.size/2)], 1):
            H[:, -ii] = np.conj(H[:, ii])
        return H

    def interpolation(self, H):
        del_bef = np.arange(self.ii_min - self.pt_interp, self.ii_min)
        del_aft = np.arange(self.ii_max, self.ii_max + self.pt_interp)
        del_elt = np.stack([del_bef, del_aft])
        Fk_inter = np.delete(self.Fk, del_elt)
        for mic in range(self.M):
            H_inter = np.delete(H[mic], del_elt)
            f = interp1d(Fk_inter, H_inter, kind='cubic')
            H[mic, :int(self.Ntfd/2)-1] = f(self.Fk)[:int(self.Ntfd/2)-1]
        return H

    def compute_beampattern(self, angle, f):
        beam = np.dot(self.filter_vector(f), self.steering_vector(angle, f))
        return np.abs(beam)

    def compute_impulse_responses(self):
        h = np.real(fft.ifft(np.conj(self.filter_matrix())))
        h = np.roll(h, int(h.size/2), axis=1)
        return h
