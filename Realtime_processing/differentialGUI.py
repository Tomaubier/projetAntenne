import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from PyQt5 import QtCore, QtGui, QtWidgets
import differentialarray as da
import scipy.signal as sg
import pyqtgraph as pg
import numpy as np
import warnings
import pyaudio
import time

pg.setConfigOptions(antialias=True)
pg.setConfigOptions(useOpenGL=True)

"""
    Interface graphique créée pour le traitement différentiel d'une antenne circulaire uniforme en temps réel.
    Réalisée par Tom Aubier et Raphaël Dumas.
"""


class BeampatternPlot(da.DifferentialArray):
    """
     Classe permettant de définir le motif de faisceau sur l'interface graphique pour ensuite pouvoir le changer avec la classe BeampatternViewBox.
    """

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
        """
        Fonction permettant d'obtenir le motif de faisceau pour une fréquence de 500 Hz et pouvoir le représenter au sein de l'interface graphique.
        """
        beam = da.DifferentialArray(steering_mic=steering_mic)
        resolution = 500
        theta = np.linspace(0, 2*np.pi, resolution)
        return (beam.beampattern(theta, f=500), theta)

    def drawHandles(self):
        """
        Fonction permettant d'obtenir de montrer les différents angles de pilotage de l'antenne différents au sein de l'interface graphique avec des ronds rouges.
        """
        self.beamPatternHandles = self.beampatternPltWidget.plot(*self.polar2cartesian(np.array([1.2]), 0), pen=None, symbol='o', symbolBrush=(250, 0, 0), symbolPen='w')
        pass

    def cartesian2polar(self, x, y, origin=(0, 0)):
        """
        Fonction permettant de passer de la base cartésienne à la base polaire.
            Entrée :
                x : 1D np.array, vecteur position suivant x
                y : 1D np.array, vecteur position suivant y
            Sortie :
                r : 1D np.array, vecteur position suivant le rayon
                theta : 1D np.array, vecteur position suivant l'angle
        """
        x, y = x - origin[0], y - origin[1]
        try:
            theta, r = np.nan_to_num(2*np.arctan(y/(x + np.sqrt(x**2 + y**2)))), np.sqrt(x**2 + y**2)
        except:
            pass
        return r, theta

    def polar2cartesian(self, r, theta):
        """
        Fonction permettant de passer de la base polaire à la base cartésienne.
            Entrée :
                r : 1D np.array, vecteur position suivant le rayon
                theta : 1D np.array, vecteur position suivant l'angle
            Sortie :
                x : 1D np.array, vecteur position suivant x
                y : 1D np.array, vecteur position suivant y
        """
        x, y = r * np.cos(theta), r * np.sin(theta)
        return x, y

    def drawAxes(self):
        """
        Fonction permettant la représentation du motif de faisceau ainsi que ses différentes variations possibles au sein de l'interface graphique.
        """
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
        """
        Fonction permettant la mise à jour du motif de faisceau en fonction de son angle de pilotage.
            Entrée :
                steering_mic : int, numéro du microphone pour l'orientation du motif de faisceau
        """
        self.beamPatternHandles.setData(*self.polar2cartesian(np.array([1.2]), (steering_mic-1)*self.beampatternVbox.micAngleStep))
        self.beamPatternCurve.setData(*self.polar2cartesian(*self.getBeampattern(steering_mic)))

    def start(self):
        """
        Fonction qui démarre le tracé du motif de faisceau en mode interactif.
        """
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()


class BeampatternViewBox(pg.ViewBox, da.DifferentialArray):
    trigBeamPatternUpdate = QtCore.pyqtSignal(int)
    """
     Classe permettant de montrer le motif de faisceau sur l'interface graphique et de le changer.
    """

    def __init__(self):
        super(BeampatternViewBox, self).__init__()
        if self.M == 6:
            self.micAngleStep = np.pi/3
        else:
            self.micAngleStep = 2*np.pi/3
        self.steeringMic = 0

    def mouseDragEvent(self, ev):
        """
         Fonction permettant de montrer le motif de faisceau sur l'interface graphique et de le changer en le glissant dans l'interface graphique.
        """
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
                self.steeringMic = newSteeringMic
                self.trigBeamPatternUpdate.emit(newSteeringMic+1)
            ev.accept()


class ParallelProcessing(QtCore.QThread, da.DifferentialArray):
    """
     Classe permettant de réaliser le traitement de l'antenne de microphones en temps réel.
    """

    def __init__(self):
        super(ParallelProcessing, self).__init__()
        self.IR = np.zeros((self.M, 2**13, self.M))
        for possibility in range(1, self.M+1):
            impulse = da.DifferentialArray(steering_mic=possibility)
            self.IR[:, :, possibility-1] = impulse.impulse_responses()

    def run(self):
        """
        Fonction permettant de réaliser la convolution du tampon avec les réponses impulsionnelles précédemment calculées pour les microphones d'intérêts.
        """
        self.processedChunk = np.sum(sg.fftconvolve(mainUI.audio.rollingBuffer, self.IR[:, :, mainUI.beampatternVbox.steeringMic], mode='same', axes=1), axis=0)[-mainUI.audio.pAudioChunkSize:]


class AudioProcessing(QtCore.QThread, da.DifferentialArray):
    trigProcess = QtCore.pyqtSignal(bool)
    """
     Classe permettant de traiter l'audio en temps réel pour créer un motif de faisceau cardioïde d'ordre 1.
    """

    def __init__(self):
        super(AudioProcessing, self).__init__()
        self.pAudio = pyaudio.PyAudio()
        self.rollingBuffer, self.pAudioChunkSize, self.Fs = np.zeros((self.M, 16000)), 256, 8000

        self.inputSet, self.outputSet = False, False
        self.inputDeviceIndexes, self.outputDeviceIndexes = self.get_io_indexes()
        self.inputDeviceNames, self.outputDeviceNames = self.get_io_names()

    def get_device_count(self):
        """
        Fonction retournant l'ensemble des dispositifs audio connectés à l'ordinateur.
            Sortie :
                count : 1D array, liste de tous les dispositifs audio
        """
        hostInfo = self.pAudio.get_host_api_info_by_index(0)
        count = hostInfo.get('deviceCount')
        return count

    def get_device_index(self, index):
        """
        Fonction retournant l'information du dispositif audio connecté à l'ordinateur en fonction de l'index.
            Entrée :
                index : int, indice du dispositif audio
            Sortie :
                info : information sur le dispositif audio
        """
        info = self.pAudio.get_device_info_by_host_api_device_index(0, index)
        return info

    def get_io_indexes(self):
        """
        Fonction retournant la liste des indices des dispositifs audio entrant et sortant de l'ordinateur.
            Sortie :
                input, output : tuple, liste des indices des dispositifs audio entrant et sortant
        """
        inputIndexes, outputIndexes = [], []
        for index in range(0, self.get_device_count()):
            device = self.get_device_index(index)
            if device['maxInputChannels'] > 0:
                inputIndexes.append(index)
            if device['maxOutputChannels'] > 0:
                outputIndexes.append(index)
        input, output = np.array(inputIndexes), np.array(outputIndexes)
        return input, output

    def get_io_names(self):
        """
        Fonction retournant le nom des indices des dispositifs audio entrant et sortant de l'ordinateur.
            Sortie :
                inputNames, outputNames : tuple, liste des noms des dispositifs audio entrant et sortant
        """
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
        inputNames, outputNames = np.array(inputNames), np.array(outputNames)
        return inputNames, outputNames

    def setInputDevice(self, selectorIndex):
        """
        Fonction permettant de sélectionnner le dispositif audio entrant avec l'interface graphique.
        """
        if selectorIndex != 0:
            self.inputDeviceIndex = self.inputDeviceIndexes[selectorIndex-1]
            self.inputDevice = self.get_device_index(self.inputDeviceIndex)
            self.nbOfInputChannels = self.inputDevice['maxInputChannels']
            self.inputSet = True
        else:
            self.inputSet = False

    def setOutputDevice(self, selectorIndex):
        """
        Fonction permettant de sélectionnner le dispositif audio sortant avec l'interface graphique.
        """
        if selectorIndex != 0:
            self.outputDeviceIndex = self.outputDeviceIndexes[selectorIndex-1]
            self.outputDevice = self.get_device_index(self.outputDeviceIndex)
            self.nbOfOutputChannels = self.outputDevice['maxOutputChannels']
            self.outputSet = True
        else:
            self.outputSet = False

    def initStream(self):
        """
        Fonction définissant les streams audio utilisés pour l'entrée et la sortie.
        """
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
        """
        Fonction permettant de fermer les streams audio et d'arrêter le traitement en temps réel.
        """
        self.inputStream.stop_stream()
        self.inputStream.close()
        self.outputStream.stop_stream()
        self.outputStream.close()

    def getMicsToErase(self):
        """
        Fonction permettant de sélectionner les voies microphoniques à conserver pour le traitement.
            Sortie :
                mics : 1D np.array, indices des voies microphoniques conservées
        """
        channels = np.arange(8)
        if self.M == 3:
            mics = np.delete(channels, channels[2::6//self.M], None)
            return mics
        else:
            mics = np.array([0, 7])
            return mics

    def run(self):
        """
        Fonction permettant de démarrer le traitement en temps réel.
        """
        while True:
            micData = np.frombuffer(self.inputStream.read(self.pAudioChunkSize), dtype=np.float32).reshape((self.pAudioChunkSize, self.nbOfInputChannels)).T

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
        """
        Fonction permettant de nettoyer les tampons utilisés pour l'audio entrant et sortant.
        """
        if self.isRunning():
            self.closeStream()
        self.pAudio.terminate()


class MainUI(QtWidgets.QWidget, BeampatternPlot):
    """
     Classe permettant de réaliser l'interface graphique et de l'utiliser.
    """

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
        """
        Fonction permettant de définir la géométrie de la fenêtre principale de l'interface graphique.
        """
        x = screenSize.width() // 2 - self.winSize[0] // 2
        y = screenSize.height() // 2 - self.winSize[1] // 1.5
        w, h = self.winSize[0], self.winSize[1]
        self.setGeometry(x, y, w, h)

    def closeEvent(self, event):
        """
        Fonction permettant quitter l'interface graphique et de nettoyer les tampons audios entrant et sortant.
        """
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
        """
        Fonction ajoutant les différents élements graphiques de l'interface et les connectant aux différentes fonctions.
        """
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
        """
        Fonction assurant la sélection des dispositifs audios entrant et sortant choisis par l'utilisateur.
        """
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
        """
        Fonction permettant de démarrer et d'arrêter le traitement en temps réel avec l'interface graphique.
        """
        if not self.audio.isRunning():
            self.audio.initStream()
            self.audio.start()
            self.recordButton.setText('Stop')
        else:
            self.audio.terminate() # Stop thread
            self.audio.closeStream()
            self.recordButton.setText('Start')


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    app = QtWidgets.QApplication(sys.argv)
    screenSize = app.primaryScreen().size()
    mainUI = MainUI()
    sys.exit(app.exec_())
