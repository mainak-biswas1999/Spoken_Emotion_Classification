def Split(ntest, X_loc, Y_loc, X_train_saveloc, Y_train_saveloc, X_test_saveloc, Y_test_saveloc, mupath, sigmapath, printdetails):
    import numpy as np
    from featureNormalize import featNorm

    X_train = (np.loadtxt(X_loc, delimiter=',', skiprows=1))
    Y_train = (np.loadtxt(Y_loc, delimiter=',', skiprows=1))[:, 1]
    
    init_ID = 1
    labelmax = X_train[-1, 0].astype(int)
    print("The number of clips in total is ", labelmax, file=printdetails)
    
    counter = np.zeros(labelmax+1, dtype=np.int32)
    for i in range(X_train.shape[0]):
        counter[X_train[i, 0].astype(int)] = counter[X_train[i, 0].astype(int)] + 1
  
    print("min frame size ", np.min(counter[1:]), file=printdetails)
    print("max frame size ", np.max(counter[1:]), file=printdetails)
    print("Avg frame size ", np.mean(counter[1:]), file=printdetails)
    m = np.mean(counter[1:]).astype(np.int32)
    #mean frame size
    #mx370x13
    
    AllData3D = []
    for i in range(1, labelmax+1):
        presentClip = (X_train[X_train[:, 0] == i])[:, 1:]
        if presentClip.shape[0] >= m:
            presentClip = presentClip[0:m, :]
        else:
            zeropads = np.zeros(shape=((m-presentClip.shape[0]), 13))
            presentClip = np.append(presentClip, zeropads, axis=0)
        AllData3D.append(presentClip)
    AllData3D = np.stack(AllData3D)

    order_permute = list(np.random.permutation(AllData3D.shape[0]))
    AllData3D = AllData3D[order_permute, :, :]
    Y_train = Y_train[order_permute]
    
    print("Shape of the total dataset_Y: ", Y_train.shape, file=printdetails)
    print("Shape of the total dataset_X: ", AllData3D.shape, file=printdetails)
    
    AllData3D = featNorm(AllData3D, mupath, sigmapath)
    print("Normalized",file=printdetails)
    
    Y_test = Y_train[-ntest:]

    counter = np.zeros(9)
    for i in range(len(Y_test)):
        counter[Y_test[i].astype(np.int8)] = counter[Y_test[i].astype(np.int8)] + 1
    print("The number of test examples per class: ", counter[1:], file=printdetails)
    
    X_test = AllData3D[-ntest:, :, :]
    X_train = AllData3D[0:-ntest, :, :]
    Y_train = Y_train[0:-ntest]
    
    print("The shape of train X, Y: ", X_train.shape, Y_train.shape, file=printdetails)
    print("The shape of test X, Y: ", X_test.shape, Y_test.shape, file=printdetails)
    
    np.save(X_train_saveloc, X_train)
    np.save(Y_train_saveloc, Y_train)
    np.save(X_test_saveloc, X_test)
    np.save(Y_test_saveloc, Y_test)
    print("Done")