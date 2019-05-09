import os

measurementName = 'ChirpWindow'

parentPath = os.path.abspath('.')
directoryPath = os.path.join(parentPath, 'Experiments', 'Data', measurementName)

for name in os.listdir(directoryPath):
    if name.endswith(".wav"):
        angleStr = name.split('_')[0][:-3]
        newFilename = os.path.join(directoryPath, '{}deg_{}.wav'.format(angleStr, measurementName))
        filename = os.path.join(directoryPath, name)

        # os.rename(filename, newFilename) # /!\ Might be dangerous /!\
        print(name, newFilename)
        continue
    else:
        continue
