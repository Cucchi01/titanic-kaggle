import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os.path
import pandas as pd
import matplotlib.pyplot as plt

N_EPOCHS = 5000
# With the value at false first it checks if there is already a model to load
FORCE_GENERATION_NEW_MODEL = True
LR = 1e-3
PATH_DATA = "data/"


class Model(nn.Module):
    def __init__(self, numInFeatures):
        super(Model, self).__init__()
        self.hidden_layer = nn.Linear(numInFeatures, 2)
        self.hidden_activation = nn.Tanh()
        self.output_linear = nn.Linear(2, 1)
        torch.nn.init.normal_(self.output_linear.weight, mean=0, std=1e-9)

    def forward(self, input):
        hidden_t = self.hidden_layer(input)
        activated_t = self.hidden_activation(hidden_t)
        output_t = self.output_linear(activated_t)
        return self.setTo1Or0(output_t, threshold= 0.5)
    
    def setTo1Or0(self, data_t, threshold=0.5):
        for i in range(data_t.shape[0]):
            if data_t[i] <= threshold:
                data_t[i] = 0
            else:
                data_t[i] = 1
        return data_t



def main():

    (TensorInputForTraining, TensorOutputForTraining) = generateTensorForTraining()

    # model = getModel(TensorInputForTraining, TensorOutputForTraining, TensorInputForTest,
    #                  TensorOutputForTest, nameModelFile='model.pth')

    # outputStatsModel(model, TensorInputForTest,
    #                  TensorOutputForTest, saveHidWeightsToFileFlag=True)


def generateTensorForTraining():
    trainData: pd.DataFrame = loadData("trainEncoded.csv", "train.csv")
    #testData: pd.DataFrame = loadData("testEncoded.npy", "test.csv")
    
    resultTraining = trainData.loc[:, "Survived"]
    inputTraining = trainData.drop("Survived", axis= 1)
    
    TensorInputForTraining = torch.from_numpy(inputTraining.to_numpy())
    TensorResForTraining = torch.from_numpy(resultTraining.to_numpy()) 

    normalizeTensor(TensorInputForTraining)
    print(TensorInputForTraining)

    TensorResForTraining = TensorResForTraining.unsqueeze(1)
    print(TensorResForTraining)

    return (TensorInputForTraining, TensorResForTraining)


def getModel(TensorInputForTraining, TensorOutputForTraining, TensorInputForTest,
             TensorOutputForTest, nameModelFile):
    if os.path.exists("model/"+nameModelFile) and FORCE_GENERATION_NEW_MODEL == False:
        model = Model(TensorInputForTraining.shape[1])
        model.load_state_dict(torch.load("model/"+nameModelFile))
    else:
        model = generateModel(TensorInputForTraining, TensorOutputForTraining, TensorInputForTest,
                              TensorOutputForTest)
        torch.save(model.state_dict(), 'model/'+nameModelFile)

    return model


def loadData(namePdFile, nameFileData):
    if os.path.exists(PATH_DATA+namePdFile):
        data: pd.DataFrame =  pd.read_csv(PATH_DATA+namePdFile, delimiter=',', header=0)
    else:
        data: pd.DataFrame = loadDataFromFile(nameFileData)
        data.to_csv(PATH_DATA+namePdFile, index=False)
    return data


def normalizeTensor(adultData):
    n_channels = adultData.shape[1]
    for i in range(0, n_channels):
        max = torch.max(adultData[:, i])
        if max > 1e-8:
            min = torch.min(adultData[:, i])
            adultData[:, i] = (adultData[:, i] - min) / max


def generateModel(TensorInputForTraining, TensorOutputForTraining, TensorInputForTest,
                  TensorOutputForTest):
    model = Model(TensorInputForTraining.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=LR)

    trainingLoop(n_epochs=N_EPOCHS, optimizer=optimizer, model=model, loss_fn=nn.MSELoss(
    ), TensorInputForTraining=TensorInputForTraining, TensorInputForTest=TensorInputForTest,  TensorOutputForTraining=TensorOutputForTraining, TensorOutputForTest=TensorOutputForTest)

    return model


def outputStatsModel(model, TensorInputForTest, TensorOutputForTest, flagPrintTensor=False, saveHidWeightsToFileFlag=True):
    if flagPrintTensor:
        print('Actual answer', TensorOutputForTest)
        print('Ouput Model:', model(TensorInputForTest))

    outputTensor = model(TensorInputForTest)

    # set output to 0 if it is lower than 0.5 or 1. They are the only
    # two possible value for that domain, the income can only be lower(0) or greater(1) of 50k
    for i in range(outputTensor.shape[0]):
        if outputTensor[i] <= 0.5:
            outputTensor[i] = 0
        else:
            outputTensor[i] = 1

    if flagPrintTensor:
        print('Ouput model ', outputTensor)

    output = list(zip(outputTensor, TensorOutputForTest))
    over50KOutput = []
    under50KOutput = []
    numDiffUnder = 0
    numDiffOver = 0
    for singleOutput in output:
        if singleOutput[1] == 1:
            over50KOutput.append(singleOutput)
            if singleOutput[0] == 0:
                numDiffOver += 1
        else:
            under50KOutput.append(singleOutput)
            if singleOutput[0] == 1:
                numDiffUnder += 1

    diffTensor = outputTensor-TensorOutputForTest
    diffTensor = torch.abs(diffTensor)
    numWrongPrediction = int(torch.sum(diffTensor))
    numPredictions = int(outputTensor.shape[0])

    print("Number of predictions: ", numPredictions)
    print("Number of correct predictions: ", numPredictions-numWrongPrediction)
    print("Number of wrong predictions: ", numWrongPrediction)
    print("Percentage of correct predictions: ",
          (numPredictions-numWrongPrediction)/numPredictions*100)
    print("Percentage of correct predictions over 50K: ",
          (1 - numDiffOver/len(over50KOutput))*100)
    print("Percentage of correct predictions under 50K: ",
          (1 - numDiffUnder/len(under50KOutput))*100)

    listFeature = hE.getListFeatureAfterHotEnc()
    # remove "income" from the list
    listFeature.pop()
    hiddenLayerWeights = model.hidden_layer.weight.data.tolist()[0]
    weightPerFeature = list(zip(listFeature, hiddenLayerWeights))
    weightPerFeature.sort(key=lambda row: row[1], reverse=True)
    if saveHidWeightsToFileFlag:
        saveHiddenWeightsToFile(weightPerFeature)


def saveHiddenWeightsToFile(weightPerFeature):
    try:
        fw = open("results/weightPerFeature.txt", "w")
        for row in weightPerFeature:
            fw.write("Feature: %-30s Weight: %3.5f\n" %
                     (row[0].decode('UTF-8'), row[1]))

        fw.close()
    except IOError:
        print("Error writing the weights of the hidden layer to file")


def loadDataFromFile(nameFile):
    df = pd.read_csv(PATH_DATA+nameFile, header=0, quotechar='"')
    df = df.drop("Name", axis=1)
    df = pd.get_dummies(df)

    return df


def mapWords(adultData):
    education2Int = {
        b"?": b"0",
        b"Preschool": b"1",
        b"1st-4th": b"2",
        b"5th-6th": b"3",
        b"7th-8th": b"4",
        b"9th": b"5",
        b"10th": b"6",
        b"11th": b"7",
        b"12th": b"8",
        b"HS-grad": b"9",
        b"Prof-school": b"10",
        b"Assoc-acdm": b"11",
        b"Assoc-voc": b"12",
        b"Some-college": b"12",
        b"Bachelors": b"14",
        b"Masters": b"15",
        b"Doctorate": b"16",
    }
    mapIncome2Int = {
        b"<=50K": b"0",
        b">50K": b"1"
    }

    for row in adultData:
        row[3] = education2Int[row[3]]
        row[14] = mapIncome2Int[row[14]]

    return adultData


def trainingLoop(n_epochs, optimizer, model, loss_fn, TensorInputForTraining, TensorInputForTest, TensorOutputForTraining, TensorOutputForTest):
    results = np.array([["Epoch", "Training Loss", "Test Loss"]])
    for epoch in range(1, n_epochs+1):
        t_p_train = model(TensorInputForTraining)
        loss_train = loss_fn(t_p_train, TensorOutputForTraining)

        t_p_val = model(TensorInputForTest)
        loss_val = loss_fn(t_p_val, TensorOutputForTest)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        results = np.append(
            results, [[epoch, float(loss_train), float(loss_val)]], axis=0)

        if epoch == 1 or epoch % 100 == 0:
            print("Epoch {}, Training loss {}, Validation loss {},".format(
                epoch, float(loss_train), float(loss_val)))

    printGraphTrainingResults(results, n_epochs)


def printGraphTrainingResults(results, n_epochs):
    # n_epochs+1 because there are the names of the columns in the first row
    results.reshape(n_epochs+1, 3)
    index = results[1:, 0]
    data = results[1:, 1:]
    columns = results[0, 1:]
    resultDF = pd.DataFrame(
        data=data, index=index, columns=columns)
    resultDF['Training Loss'] = resultDF['Training Loss'].astype(float)
    resultDF['Test Loss'] = resultDF['Test Loss'].astype(float)
    resultDF.plot()
    plt.show()


main()