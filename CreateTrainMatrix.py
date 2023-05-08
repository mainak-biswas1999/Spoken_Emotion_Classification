def createTrainData(loc, listOfNoises, printDetails):
    from audioRead import readDirectory
    import numpy as np
    import os
    import csv
    folders = os.listdir(loc)
    TrainMatrix = []
    Y = []
    READ = 0
    for i in range(len(folders)):
        langi, nRead = readDirectory(loc+'/'+folders[i], READ, listOfNoises, printDetails)
        
        if len(TrainMatrix) == 0:
            TrainMatrix = langi
        else:
            TrainMatrix = np.append(TrainMatrix, langi, axis=0)
        
        ids = np.linspace(READ+1, READ+nRead, nRead).astype(int)
        label = np.array(np.ones(nRead)*(i+1)).astype(int);
        toappend = np.column_stack((ids, label));
        
        if len(Y) == 0:
            Y = toappend
        else:
            Y = np.append(Y, toappend, axis = 0)
            
        READ = READ + nRead
    
    return TrainMatrix, Y;
    