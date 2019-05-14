import numpy as np

C = 342
ang = 0

class PlaneWave:
    """
        Créé une onde plane arrivant sur l'antenne.
        Entrée :
            amp : amplitude de l'onde plane
            f : fréquence de l'onde plane
            ang : angle de l'onde plane
    """
    def __init__(self, amp, f, ang):
        self.amp, self.f, self.ang = amp, f, ang
        self.w = 2 * np.pi * self.f
        self.k = self.w / C * np.array([np.cos(self.ang), np.sin(self.ang)])

    def field(self, x, y, t, *t_mesh):
        """
            Créé le champ de pression de l'onde plane à tout instant aux points (x, y).
            Entrée :
                x : vecteur position suivant x (1d array)
                y : vecteur position suivant y (1d array)
                t : vecteur temps (1d array)
            Sortie :
                ndarray (x.size, t.size): champ de pression complexe
        """
        time = np.exp(1j*self.w*t)
        field = self.amp*np.exp(1j*np.dot(self.k, [x, y]))
        return np.dot(field.reshape(x.size, 1), time.reshape(1, t.size))


class CircularArray:
    """
        Créé une antenne circulaire uniforme.
        Entrée :
            M : nombre de microphones
            R : rayon de l'antenne circulaire
    """
    def __init__(self, M, R):
        self.M, self.R = M, R
        self.phi = np.arange(0, 2 * np.pi, 2 * np.pi / self.M)
        self.tau = self.R/C

    def coordinates(self):
        return self.R * np.cos(self.phi), self.R * np.sin(self.phi)

    def pattern(self, theta, f, frontMic):
        d = np.array([np.exp(-1j*2*np.pi*f*self.tau*np.cos(theta - phi)) for phi in self.phi])
        return np.abs(np.dot(self.filter_order3(f, frontMic), d))

    def filter_order3(self, f):
        A = np.zeros((self.M, self.M), dtype='cfloat')
        b = np.zeros(self.M, dtype='cfloat')
        b[0] = 1

        theta_c = [0, np.pi/2, 2*np.pi/3, np.pi]
        for ii, angle in enumerate(theta_c):
            A[ii] = np.array([np.exp(-1j*2*np.pi* f * self.tau*np.cos(angle - phi)) for phi in self.phi], dtype='cfloat')
        A[-2] = np.array([0, 1 , 0, 0, 0, -1], dtype='cfloat')
        A[-1] = np.array([0, 0, 1 ,0, -1, 0], dtype='cfloat')
        h = np.linalg.solve(A, b)
        return h

    def filter_order3(self, f, frontMic):
        A = np.zeros((self.M, self.M), dtype='cfloat')
        b = np.zeros(self.M, dtype='cfloat')
        b[frontMic] = 1

        theta_c = np.array([0, np.pi/2, 2*np.pi/3, np.pi])
        theta_c += frontMic*np.pi/3
        for ii, angle in enumerate(theta_c):
            A[ii] = np.array([np.exp(-1j*2*np.pi* f * self.tau*np.cos(angle - phi)) for phi in self.phi], dtype='cfloat')
        A[-2] = np.roll(np.array([0, 1 , 0, 0, 0, -1], dtype='cfloat'), frontMic)
        A[-1] = np.roll(np.array([0, 0, 1 ,0, -1, 0], dtype='cfloat'), frontMic)
        h = np.linalg.solve(A, b)
        return h


def freq_aliasing(x, y):
    """
        Calcul la fréquence de repliement en fonction de la distance entre les Microphones
        (Sous-estimée pour l'instant)
    """
    delta = np.sqrt(np.abs(x[0] - x[1])**2 + np.abs(y[0] - y[1])**2)
    return C/(2*delta)



def index(Fk, fmax):
    """
        Retourne l'indice d'un axe Fk le plus proche de la fréquence fmax
        (Existe deja avec numpy ?)
    """
    ii = 0
    while Fk[ii] < fmax: ii += 1
    return ii-1
