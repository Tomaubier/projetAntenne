from PyQt5 import QtCore, QtGui, QtWidgets
import differentialarrayFirstOrder as da # differentialarrayFirstOrder / differentialarrayThirdOrder
import scipy.signal as sg
import pyqtgraph as pg
import numpy as np
import warnings
import pyaudio
import time
import sys

pg.setConfigOptions(antialias=True)
pg.setConfigOptions(useOpenGL=True)

class AudioProcessing(QtCore.QThread, da.DifferentialArray):
    trigProcess = QtCore.pyqtSignal(bool)
    def __init__(self):
        super(AudioProcessing, self).__init__()
        self.pAudio = pyaudio.PyAudio()
        self.rollingBuffer, self.pAudioChunkSize, self.Fs = np.zeros((self.M, 16000)), 128, 8000

        self.inputSet, self.outputSet = False, False
        self.inputDeviceIndexes, self.outputDeviceIndexes = self.get_io_indexes()
        self.inputDeviceNames, self.outputDeviceNames = self.get_io_names()

    def get_device_count(self):
        hostInfo = self.pAudio.get_host_api_info_by_index(0)
        return hostInfo.get('deviceCount')

    def get_device_index(self, index):
        return self.pAudio.get_device_info_by_host_api_device_index(0, index)

    def get_io_indexes(self):
        inputIndexes, outputIndexes = [], []
        for index in range(0, self.get_device_count()):
            device = self.get_device_index(index)
            if device['maxInputChannels'] > 0:
                inputIndexes.append(index)
            if device['maxOutputChannels'] > 0:
                outputIndexes.append(index)
        return np.array(inputIndexes), np.array(outputIndexes)

    def get_io_names(self):
        inputNames, outputNames = [], []
        for index in range(0, self.get_device_count()):
            device = self.get_device_index(index)
            if len(device['name'].split(' ')) < 3:
                device_name = device['name']
            else:
                device_name = ' '.join(device['name'].split(' ')[:3])
            if device['maxInputChannels'] > 0:
                inputNames.append('{} ({} Ch.)'.format(device_name, device['maxInputChannels']))
            if device['maxOutputChannels'] > 0:
                outputNames.append('{} ({} Ch.)'.format(device_name, device['maxOutputChannels']))
        return np.array(inputNames), np.array(outputNames)

    def setInputDevice(self, selectorIndex):
        if selectorIndex != 0:
            self.inputDeviceIndex = self.inputDeviceIndexes[selectorIndex-1]
            self.inputDevice = self.get_device_index(self.inputDeviceIndex)
            self.nbOfInputChannels = self.inputDevice['maxInputChannels']
            self.inputSet = True
        else:
            self.inputSet = False

    def setOutputDevice(self, selectorIndex):
        if selectorIndex != 0:
            self.outputDeviceIndex = self.outputDeviceIndexes[selectorIndex-1]
            self.outputDevice = self.get_device_index(self.outputDeviceIndex)
            self.nbOfOutputChannels = self.outputDevice['maxOutputChannels']
            self.outputSet = True
        else:
            self.outputSet = False

    def initStream(self):
        self.micsToErase = self.getMicsToErase()
        self.inputStream = self.pAudio.open(format = pyaudio.paFloat32,
                        frames_per_buffer = self.pAudioChunkSize,
                        rate = self.Fs,
                        input = True,
                        input_device_index = self.inputDeviceIndex,
                        channels = self.nbOfInputChannels)
        self.outputStream = self.pAudio.open(format = pyaudio.paFloat32,
                        frames_per_buffer = self.pAudioChunkSize,
                        rate = self.Fs,
                        input = False,
                        output = True,
                        output_device_index = self.outputDeviceIndex,
                        channels = self.nbOfOutputChannels)

    def closeStream(self):
        self.inputStream.stop_stream()
        self.inputStream.close()
        self.outputStream.stop_stream()
        self.outputStream.close()

    def getMicsToErase(self):
        channels = np.arange(8)
        if self.M == 3:
            mics = np.delete(channels, channels[2::6//self.M], None)
            return mics
        else:
            return np.array([0, 7])

    def run(self):
        while True:
            try:
                rawData = np.frombuffer(self.inputStream.read(self.pAudioChunkSize), dtype=np.float32)
                micData = rawData.reshape((self.pAudioChunkSize, self.nbOfInputChannels)).T
            except:
                print('Input Overflow')
                micData = np.zeros((self.nbOfInputChannels, self.pAudioChunkSize))

            self.rollingBuffer = np.roll(self.rollingBuffer, -self.pAudioChunkSize)
            self.rollingBuffer[:, -self.pAudioChunkSize:] = np.delete(micData, self.micsToErase, axis=0)
            self.trigProcess.emit(True)
            try:
                outputData = mainUI.parallelProcess.processedChunk
            except:
                outputData = np.zeros(self.pAudioChunkSize)
            outputData = np.repeat(outputData, self.nbOfOutputChannels, axis=0)
            self.outputStream.write(outputData.astype(np.float32).tostring())

    def clean(self):
        if self.isRunning():
            self.closeStream()
        self.pAudio.terminate()


class BeampatternPlot(da.DifferentialArray):
    def __init__(self):
        super(BeampatternPlot, self).__init__()

        self.beampatternVbox = BeampatternViewBox()
        self.beampatternPltWidget = pg.PlotWidget(viewBox=self.beampatternVbox)
        self.beampatternPltWidget.setAspectLocked()
        self.beampatternPltWidget.setMouseEnabled(x=False, y=False)
        self.beampatternPltWidget.hideAxis('bottom')
        self.beampatternPltWidget.hideAxis('left')
        self.beampatternPltWidget.plot(*self.polar2cartesian(np.array([1.2]), np.arange(0, 5*np.pi/3, np.pi/3)), pen=None, symbol='o', symbolBrush=(50, 50, 50), symbolPen='w')
        self.beamPatternCurve = self.beampatternPltWidget.plot(pen=pg.mkPen(width=3, color=(255, 153.0, 7.65)))
        self.beampatternVbox.trigBeamPatternUpdate.connect(self.plotBeampattern)

        self.drawAxes()
        self.drawHandles()
        self.plotBeampattern(1)

    def getBeampattern(self, steering_mic):
        beam = da.DifferentialArray(steering_mic=steering_mic)
        resolution = 500
        theta = np.linspace(0, 2*np.pi, resolution)
        return (beam.compute_beampattern(theta, f=500), theta)

    def drawHandles(self):
        self.beamPatternHandles = self.beampatternPltWidget.plot(*self.polar2cartesian(np.array([1.2]), 0), pen=None, symbol='o', symbolBrush=(250, 0, 0), symbolPen='w')
        pass

    def cartesian2polar(self, x, y, origin=(0, 0)):
        x, y = x - origin[0], y - origin[1]
        try:
            theta, r = np.nan_to_num(2*np.arctan(y/(x + np.sqrt(x**2 + y**2)))), np.sqrt(x**2 + y**2)
        except:
            pass
        return r, theta

    def polar2cartesian(self, r, theta):
        x, y = r * np.cos(theta), r * np.sin(theta)
        return x, y

    def drawAxes(self):
        rMax = 1.2
        # Add polar grid lines
        self.beampatternPltWidget.plot(*self.polar2cartesian(np.array([-rMax, rMax]), np.pi/3), pen=0.3)
        self.beampatternPltWidget.plot(*self.polar2cartesian(np.array([-rMax, rMax]), 0), pen=0.3)
        self.beampatternPltWidget.plot(*self.polar2cartesian(np.array([-rMax, rMax]), 2*np.pi/3), pen=0.3)
        for r in np.arange(0, rMax, 0.2):
            circle = pg.QtGui.QGraphicsEllipseItem(-r, -r, r*2, r*2)
            circle.setPen(pg.mkPen(0.2))
            self.beampatternPltWidget.addItem(circle)
        circle = pg.QtGui.QGraphicsEllipseItem(-rMax, -rMax, rMax*2, rMax*2)
        circle.setPen(pg.mkPen(0.5))
        self.beampatternPltWidget.addItem(circle)

    def plotBeampattern(self, steering_mic):
        self.beamPatternHandles.setData(*self.polar2cartesian(np.array([1.2]), (steering_mic-1)*self.beampatternVbox.micAngleStep))
        self.beamPatternCurve.setData(*self.polar2cartesian(*self.getBeampattern(steering_mic)))

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()


class BeampatternViewBox(pg.ViewBox, da.DifferentialArray):
    trigBeamPatternUpdate = QtCore.pyqtSignal(int)
    def __init__(self):
        super(BeampatternViewBox, self).__init__()
        if self.M == 6:
            self.micAngleStep = np.pi/3
        else:
            self.micAngleStep = 2*np.pi/3
        self.steeringMic = 1

    def mouseDragEvent(self, ev):
        if (ev.button() == QtCore.Qt.LeftButton):
            pg.ViewBox.mouseDragEvent(self, ev)
            xMouse, yMouse = self.mapToView(ev.pos()).x(), self.mapToView(ev.pos()).y()
            rMouse, thetaMouse = mainUI.beampatternPlot.cartesian2polar(xMouse, yMouse)
            newSteeringMic = int((thetaMouse+self.micAngleStep/2)//self.micAngleStep)
            if self.M == 6:
                newSteeringMic = (newSteeringMic if newSteeringMic >= 0 else newSteeringMic+6)
            else:
                newSteeringMic = (newSteeringMic if newSteeringMic >= 0 else 2)
            if newSteeringMic != self.steeringMic:
                print(newSteeringMic)
                self.steeringMic = newSteeringMic
                self.trigBeamPatternUpdate.emit(newSteeringMic+1)
            ev.accept()


class MainUI(QtWidgets.QWidget, BeampatternPlot):
    def __init__(self):
        super().__init__()
        self.winSize = (600, 600)
        self.setWindowTitle('Real-Time Differential Array Beamforming')
        self.geometry()
        self.audio, self.parallelProcess = AudioProcessing(), ParallelProcessing()
        self.audio.trigProcess.connect(self.parallelProcess.start)

        self.beampatternPlot = BeampatternPlot()
        self.ui_elements()
        self.show()

    def geometry(self):
        x = screenSize.width() // 2 - self.winSize[0] // 2
        y = screenSize.height() // 2 - self.winSize[1] // 1.5
        w, h = self.winSize[0], self.winSize[1]
        self.setGeometry(x, y, w, h)

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

    def ui_elements(self):
        ## Buttons
        self.applyAudioSettingsButton = QtWidgets.QPushButton('Apply Settings', self)
        self.applyAudioSettingsButton.clicked.connect(self.applyAudioSettings)

        self.recordButton = QtWidgets.QPushButton('Start', self)
        self.recordButton.setEnabled(False)
        self.recordButton.clicked.connect(self.startStopRecording)

        ## QComboBox
        self.inputDeviceSelector = QtWidgets.QComboBox(self)
        self.inputDeviceSelector.addItem('Select Input Device')
        for each in self.audio.inputDeviceNames:
            self.inputDeviceSelector.addItem(each)
        self.inputDeviceSelector.activated.connect(self.audio.setInputDevice)

        self.outputDeviceSelector = QtWidgets.QComboBox(self)
        self.outputDeviceSelector.addItem('Select Output Device')
        for each in self.audio.outputDeviceNames:
            self.outputDeviceSelector.addItem(each)
        self.outputDeviceSelector.activated.connect(self.audio.setOutputDevice)

        ## QLabel
        self.audioSettingsLabel = QtWidgets.QLabel(self)
        self.audioSettingsLabel.setText('Audio Settings')

        ## QAction
        self.quit = QtWidgets.QAction("Quit", self)
        self.quit.triggered.connect(self.closeEvent)

        # Beampattern
        # self.beampatternPltWidget.setEnabled(False)

        ## Layouts
        self.audioSettingsLayout = QtWidgets.QGridLayout()
        self.audioSettingsLayout.addWidget(self.audioSettingsLabel, 0, 0)
        self.audioSettingsLayout.addWidget(self.inputDeviceSelector, 1, 0)
        self.audioSettingsLayout.addWidget(self.outputDeviceSelector, 1, 1)

        self.recordingLayout = QtWidgets.QGridLayout()
        self.recordingLayout.addWidget(self.recordButton, 2, 0)

        masterLayout = QtWidgets.QGridLayout()
        self.setLayout(masterLayout)
        masterLayout.addLayout(self.audioSettingsLayout, 0, 0)
        masterLayout.addWidget(self.applyAudioSettingsButton, 1, 0)
        masterLayout.addWidget(self.beampatternPltWidget, 2, 0)
        masterLayout.addLayout(self.recordingLayout, 3, 0)

    def applyAudioSettings(self):
        if self.audio.inputSet & self.audio.outputSet:
            if self.audio.nbOfInputChannels == 8:
                self.applyAudioSettingsButton.setEnabled(False)
                self.inputDeviceSelector.setEnabled(False)
                self.outputDeviceSelector.setEnabled(False)
                self.beampatternPltWidget.setEnabled(True)
                self.recordButton.setEnabled(True)
            else:
                QtWidgets.QMessageBox.information(self, 'Info', 'An input device with 8 channels is required...', QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        elif not self.audio.inputSet:
            QtWidgets.QMessageBox.information(self, 'Info', 'No Input Selected...', QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
        elif not self.audio.outputSet:
            QtWidgets.QMessageBox.information(self, 'Info', 'No Output Selected...', QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)

    def startStopRecording(self):
        if not self.audio.isRunning():
            self.audio.initStream()
            self.audio.start()
            self.recordButton.setText('Stop')
        else:
            self.audio.terminate() # Stop thread
            self.audio.closeStream()
            self.recordButton.setText('Start')


class ParallelProcessing(QtCore.QThread, da.DifferentialArray):
    def __init__(self):
        super(ParallelProcessing, self).__init__()
        self.IR = np.zeros((self.M, 2**13, self.M))
        for possibility in range(1, self.M+1):
            impulse = da.DifferentialArray(steering_mic=possibility)
            self.IR[:, :, possibility-1] = impulse.compute_impulse_responses()

    def run(self):
        self.processedChunk = np.sum(sg.fftconvolve(mainUI.audio.rollingBuffer, self.IR[:, :, 0], mode='same', axes=1)[:, -mainUI.audio.pAudioChunkSize:], axis=0)


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    app = QtWidgets.QApplication(sys.argv)
    screenSize = app.primaryScreen().size()
    mainUI = MainUI()
    sys.exit(app.exec_())