from PyQt5 import QtCore, QtGui, QtWidgets
from scipy.interpolate import interp1d
import scipy.signal as sg
import numpy.fft as fft
import numpy as np
import pyaudio
import time
import sys


class MainUI(QtWidgets.QWidget):
    def __init__(self):
        super(MainUI, self).__init__()
        self.winSize = (600, 200)
        self.setGeometry(screenSize.width() // 2 - self.winSize[0] // 2, screenSize.height() // 2 - self.winSize[1], self.winSize[0], self.winSize[1])
        self.setWindowTitle('Mic Array Real Time Beamforming')

        self.audio = Audio()
        self.antenne = CircularArray(3, 4.32e-2)
        self.parallelProcess = ParallelProcessing()
        self.inputSet = False
        self.outputSet = False
        self.inputDeviceIndexes = []
        self.outputDeviceIndexes = []
        self.audio.trigProcess.connect(self.parallelProcess.start)
        self.uiElements()
        self.quit = QtWidgets.QAction("Quit", self)
        self.quit.triggered.connect(self.closeEvent)
        self.show()

    def closeEvent(self, event):
        closeMessage = QtWidgets.QMessageBox()
        closeMessage.setText('Quit Application?')
        closeMessage.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        closeMessage = closeMessage.exec()
        if closeMessage == QtWidgets.QMessageBox.Yes:
            self.audio.clean()
            event.accept()
        else:
            event.ignore()

    def uiElements(self):
        masterLayout = QtWidgets.QGridLayout()
        self.setLayout(masterLayout)

        audioSettingsLayout = QtWidgets.QGridLayout()
        masterLayout.addLayout(audioSettingsLayout, 0, 0)
        self.recordingLayout = QtWidgets.QGridLayout()
        masterLayout.addLayout(self.recordingLayout, 2, 0)

        # Audio Settings:

        audioSettingsLabel = QtWidgets.QLabel(self)
        audioSettingsLabel.setText('Audio Settings')
        audioSettingsLayout.addWidget(audioSettingsLabel, 0, 0)

        self.inputDeviceSelector = QtWidgets.QComboBox(self)
        self.inputDeviceSelector.addItem('Select Input Device')
        for deviceIndex in range(0, self.audio.deviceCount):
            device = self.audio.pAudio.get_device_info_by_host_api_device_index(0, deviceIndex)
            if device['maxInputChannels'] > 0:
                self.inputDeviceIndexes.append(deviceIndex)
                self.inputDeviceSelector.addItem('{} ({} Ch.)'.format(device['name'] if len(device['name'].split(' ')) < 3 else ' '.join(device['name'].split(' ')[:3]), device['maxInputChannels']))
        audioSettingsLayout.addWidget(self.inputDeviceSelector, 1, 0)
        self.inputDeviceSelector.activated.connect(self.setInputDevice)

        self.outputDeviceSelector = QtWidgets.QComboBox(self)
        self.outputDeviceSelector.addItem('Select Output Device')
        for deviceIndex in range(0, self.audio.deviceCount):
            device = self.audio.pAudio.get_device_info_by_host_api_device_index(0, deviceIndex)
            if device['maxOutputChannels'] > 0:
                self.outputDeviceIndexes.append(deviceIndex)
                self.outputDeviceSelector.addItem('{} ({} Ch.)'.format(device['name'] if len(device['name'].split(' ')) < 3 else ' '.join(device['name'].split(' ')[:3]), device['maxOutputChannels']))
        audioSettingsLayout.addWidget(self.outputDeviceSelector, 1, 1)
        self.outputDeviceSelector.activated.connect(self.setOutputDevice)

        self.applyAudioSettingsButton = QtWidgets.QPushButton('Apply Settings', self)
        self.applyAudioSettingsButton.clicked.connect(self.applyAudioSettings)
        masterLayout.addWidget(self.applyAudioSettingsButton, 1, 0)


        self.infosLabel = QtWidgets.QLabel(self)
        self.infosLabel.setText('Motif de faisceau du 1er ordre | Orientaion : 0°')
        self.recordingLayout.addWidget(self.infosLabel, 0, 0)

        self.orientationSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.orientationSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.orientationSlider.setMinimum(1)
        self.orientationSlider.setMaximum(3)
        self.orientationSlider.setValue(2)
        self.orientationSlider.valueChanged[int].connect(self.updateBeampatternOrientation)
        self.recordingLayout.addWidget(self.orientationSlider, 1, 0)

        self.recordButton = QtWidgets.QPushButton('Start', self)
        self.recordButton.clicked.connect(self.startStopRecording)
        self.recordingLayout.addWidget(self.recordButton, 2, 0)

        self.recordButton.setEnabled(False)
        self.orientationSlider.setEnabled(False)
        self.infosLabel.setEnabled(False)


    def updateBeampatternOrientation(self, sliderValue):
        self.infosLabel.setText('Motif de faisceau du 1er ordre | Orientaion : {}°'.format(sliderValue))

    def setInputDevice(self, selectorIndex):
        if selectorIndex != 0:
            self.inputDeviceIndex = self.inputDeviceIndexes[selectorIndex-1]
            self.inputDevice = self.audio.pAudio.get_device_info_by_host_api_device_index(0, self.inputDeviceIndex)
            self.nbOfInputChannels = self.inputDevice['maxInputChannels']
            self.inputSet = True
        else:
            self.inputSet = False

    def setOutputDevice(self, selectorIndex):
        if selectorIndex != 0:
            self.outputDeviceIndex = self.outputDeviceIndexes[selectorIndex-1]
            self.outputDevice = self.audio.pAudio.get_device_info_by_host_api_device_index(0, self.outputDeviceIndex)
            self.nbOfOutputChannels = self.outputDevice['maxOutputChannels']
            self.outputSet = True
        else:
            self.outputSet = False

    def applyAudioSettings(self):
        if self.inputSet & self.outputSet:
            if self.nbOfInputChannels == 8:
                self.applyAudioSettingsButton.setEnabled(False)
                self.inputDeviceSelector.setEnabled(False)
                self.outputDeviceSelector.setEnabled(False)
                self.orientationSlider.setEnabled(True)
                self.recordButton.setEnabled(True)
                self.infosLabel.setEnabled(True)
            else:
                QtWidgets.QMessageBox.information(self, 'Info', 'An input device with 8 channels is required...', QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        elif not self.inputSet:
            QtWidgets.QMessageBox.information(self, 'Info', 'No Input Selected...', QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        elif not self.outputSet:
            QtWidgets.QMessageBox.information(self, 'Info', 'No Output Selected...', QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    def startStopRecording(self):
        if not self.audio.isRunning(): # Is thread running?
            self.audio.initStream()
            self.audio.start()
            self.recordButton.setText('Stop')
        else:
            self.audio.terminate() # Stop thread
            self.audio.closeStream()
            self.recordButton.setText('Start')


class CircularArray:
    """
        Créé une antenne circulaire uniforme.
        Entrée :
            M : nombre de microphones
            R : rayon de l'antenne circulaire
    """
    def __init__(self, M, R):
        self.C = 342
        self.M, self.R = M, R
        self.phi = np.arange(0, 2 * np.pi, 2 * np.pi / self.M)
        self.tau = self.R/self.C
        self.Fs, self.Ntfd = 8192, 2**13
        self.Fk = np.arange(self.Ntfd) * self.Fs / self.Ntfd
        self.filterGains = self.computeGains()
        self.filterGainsConvolution = self.computeGains()[:, 3700:-3700] # For convolution only!!! ;)

    def coordinates(self):
        return self.R * np.cos(self.phi), self.R * np.sin(self.phi)

    def pattern(self, theta, f):
        d = np.array([np.exp(-1j*2*np.pi*f*self.tau*np.cos(theta - phi)) for phi in self.phi])
        return np.abs(np.dot(self.filter_order1(f), d))

    def filter_order1(self, f):
        A = np.zeros((self.M, self.M), dtype='cfloat')
        b = np.zeros(self.M, dtype='cfloat')
        b[0] = 1

        theta_c = [0, np.pi]
        for ii, angle in enumerate(theta_c):
            A[ii] = np.array([np.exp(-1j*2*np.pi* f * self.tau*np.cos(angle - phi)) for phi in self.phi], dtype='cfloat')
        A[-1] = np.array([0, 1 ,-1], dtype='cfloat')
        h = np.linalg.solve(A, b)
        return h

    def index(self, Fk, fmax):
        ii = 0
        while Fk[ii] < fmax: ii += 1
        return ii

    def computeGains(self, frequencyBand=(441, 1851)):
        """Calcul des gains dans la bande de fréquences [frequencyBand[0], frequencyBand[1] ["""
        ii_min, ii_max = self.index(self.Fk, frequencyBand[0]), self.index(self.Fk, frequencyBand[1])
        H = np.zeros((self.M, self.Fk.size), dtype='cfloat')
        for ii, freq in enumerate(self.Fk[ii_min:ii_max], ii_min):
            H[:, ii] = self.filter_order1(freq)

        pt_interp = 10 # Nb de points de l'interpolation
        del_bef, del_aft = np.arange(ii_min - pt_interp, ii_min),  np.arange(ii_max, ii_max + pt_interp)
        del_elt = np.stack([del_bef, del_aft]) # Elements à supprimer pour l'interpolation
        # Interpolation
        Fk_inter = np.delete(self.Fk, del_elt)
        for mic in range(self.M):
            H_inter = np.delete(H[mic], del_elt)
            f = interp1d(Fk_inter, H_inter, kind='cubic')
            H[mic, :int(self.Ntfd/2)-1] = f(self.Fk)[:int(self.Ntfd/2)-1]
        # Symétrie Hermitienne
        for ii, freq in enumerate(self.Fk[1:int(self.Fk.size/2)], 1):
            H[:, -ii] = np.conj(H[:, ii])
        # Réponse impulsionnelle h
        h = np.real(fft.ifft(np.conj(H)))
        # IR non causale vers causale (equivalent a circshift sous Matlab)
        h = np.roll(h, int(h.size/2), axis=1)
        return h


class ParallelProcessing(QtCore.QThread):
    def __init__(self):
        super(ParallelProcessing, self).__init__()
        self.timeStart = time.time()
        self.previousTime = 0
        self.cutOffOffset = 0.5
        # self.processedChunkLock = QtCore.QMutex()
        # self.processedChunkLock.lock()

    def run(self):
        # print(mainUI.audio.rollingBuffer.shape)
        # print('{:.2f}\t{:.2f}\t{:.2f}'.format(*np.max(mainUI.audio.rollingBuffer, axis=1)))
        # self.processedChunk = np.mean(mainUI.audio.rollingBuffer, axis=0)[-396-mainUI.audio.pAudioChunkSize:-396] # output the mean of the 3 mics
        self.processedChunk = np.sum(sg.fftconvolve(mainUI.audio.rollingBuffer, mainUI.antenne.filterGainsConvolution, mode='same', axes=1), axis=0)[-mainUI.audio.pAudioChunkSize:]


class Audio(QtCore.QThread):
    trigProcess = QtCore.pyqtSignal(bool)
    def __init__(self):
        super(Audio, self).__init__()
        self.pAudio = pyaudio.PyAudio()
        hostInfo = self.pAudio.get_host_api_info_by_index(0)
        self.deviceCount = hostInfo.get('deviceCount')

        self.pAudioFormat = pyaudio.paFloat32 # paFloat32
        self.npProcessingFormat = np.float32
        self.pAudioSampleRate = 8192
        self.pAudioChunkSize = 256
        self.rollingBufferSize = 16000

    def initStream(self):
        self.rollingBuffer = np.zeros((3, self.rollingBufferSize))
        self.inputStream = self.pAudio.open(format = self.pAudioFormat,
                        rate = self.pAudioSampleRate,
                        frames_per_buffer = self.pAudioChunkSize,
                        input = True,
                        input_device_index = mainUI.inputDeviceIndex,
                        channels = mainUI.nbOfInputChannels)
        self.outputStream = self.pAudio.open(format = self.pAudioFormat,
                        rate = self.pAudioSampleRate,
                        frames_per_buffer = self.pAudioChunkSize,
                        input = False,
                        output = True,
                        output_device_index = mainUI.outputDeviceIndex,
                        channels = mainUI.nbOfOutputChannels)

    def closeStream(self):
        self.inputStream.stop_stream()
        self.inputStream.close()
        self.outputStream.stop_stream()
        self.outputStream.close()

    def run(self):
        while True:
            micData = np.frombuffer(self.inputStream.read(self.pAudioChunkSize), dtype=self.npProcessingFormat).reshape((self.pAudioChunkSize, mainUI.nbOfInputChannels)).T
            # 1st Channel: Centered mic
            # Keep only 3 mics (1, 3, 5) for now

            self.rollingBuffer = np.roll(self.rollingBuffer, -self.pAudioChunkSize)
            self.rollingBuffer[:, -self.pAudioChunkSize:] = np.delete(micData, [0, 1, 3, 5, 7], axis=0)

            self.trigProcess.emit(True)

            # outputData = np.sum(sg.fftconvolve(micData, mainUI.antenne.filterGains, mode='full', axes=1), axis=0)
            # currentTime = time.time() - self.timeStart
            # print(currentTime - self.previousTime)
            # self.previousTime = currentTime

            try:
                outputData = mainUI.parallelProcess.processedChunk
            except:
                outputData = np.zeros(self.pAudioChunkSize)
                # print('Zeros')

            # outputData = np.mean(micData, axis=0) # output the mean of the 3 mics
            outputData = np.repeat(outputData, mainUI.nbOfOutputChannels, axis=0) # repeat for each output channels
            self.outputStream.write(outputData.astype(self.npProcessingFormat).tostring())

    def clean(self):
        if self.isRunning():
            self.closeStream()
        self.pAudio.terminate()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    screenSize = app.primaryScreen().size()
    mainUI = MainUI()
    sys.exit(app.exec_())

# Increase nb of points to process
# self.rollingBuffer = np.roll(self.rollingBuffer, self.pAudioChunkSize, axis=1)
# self.rollingBuffer[:, -self.pAudioChunkSize:] = micData
#
# outputData = np.real(np.fft.ifft(np.fft.fft(self.rollingBuffer[0]))) # Processing

# outputData = np.repeat(outputData[-self.pAudioChunkSize:], mainUI.nbOfOutputChannels, axis=0)

# Visualise mics max values
# print('{:.2f}\t{:.2f}\t{:.2f}'.format(*np.max(micData, axis=1)))
