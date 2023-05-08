def readDirectory(Directory, startIndex, listOfNoises, printDetails): 
    import numpy as np
    from glob import glob
    import librosa as lib
    #import DataAugmentation as DAug
    
    #setting data directory
    audio_files = glob(Directory+'/*.wav')
    print("reading ",len(audio_files)," files from ",Directory, file=printDetails)
    print("reading ",len(audio_files)," files from ",Directory)
    Xdata = []
    #reading the files
    #for i in range(10):
    for i in range(len(audio_files)):
        y, fs = lib.load(audio_files[i], sr = None);
        windz = int(fs*0.025);   #window size
        shift = int(fs*0.010);
        
        #Y = DAug.addNoise(y, listOfNoises)
        Y = y
        
        mfcc = lib.feature.mfcc(y=Y, sr=fs, n_mfcc = 14, hop_length = shift, n_fft = windz)
        mfcc = mfcc[1:14, :]
        
        ids = np.array([(1+i)*np.ones(mfcc.shape[1]) + startIndex]).astype(int)
        mfcc = np.append(ids, mfcc, axis = 0).T;
        
        if len(Xdata) == 0:
            Xdata = mfcc
        else:
            Xdata = np.append(Xdata, mfcc, axis=0)
    
    return Xdata, len(audio_files);
    #return Xdata, 10
