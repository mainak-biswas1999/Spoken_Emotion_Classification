#def Caller(dataloc, noiseloc):
def Caller(dataloc, noiseloc, X_train_saveloc, Y_train_saveloc, X_test_saveloc, Y_test_saveloc):
    from CreateTrainMatrix import createTrainData
    import csv
    import pandas as pd
    import SplitExampleSet as se
    #from GetListOfNoises import getListOfNoises
    
    muMFCC = "CSVFiles/Stats/muMFCC.csv"
    sigmaMFCC = "CSVFiles/Stats/sigmaMFCC.csv"
    DataX = "CSVFiles/AllClipsMFCC/DataAllX_notnorm.csv"  
    DataY = "CSVFiles/AllClipsMFCC/DataAllY_notnorm.csv"
    printDetails = open('CSVFiles/ExtractionDetails.txt', 'w')
    #listOfNoises = getListOfNoises(noiseloc)
    listOfNoises = []
    
    Data, Y = createTrainData(dataloc, listOfNoises, printDetails)
    
    pd.DataFrame(Data).to_csv(DataX, index = False)
    pd.DataFrame(Y).to_csv(DataY, index = False)
    #end of use 1#
    printDetails.close()
    
    printDetails = open('CSVFiles/SplitDetails.txt', 'w')
    se.Split(200, DataX, DataY, X_train_saveloc, Y_train_saveloc, X_test_saveloc, Y_test_saveloc, muMFCC, sigmaMFCC, printDetails)
    printDetails.close()