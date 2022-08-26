import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os.path
import pandas as pd
import matplotlib.pyplot as plt

# import sys
# print("Python version: {}". format(sys.version))
# print("pandas version: {}". format(pd.__version__))
# print("NumPy version: {}". format(np.__version__))
# print("PyTorch version: {}". format(torch.__version__))
# import matplotlib
# print('Matplotlib: {}'.format(matplotlib.__version__))

N_EPOCHS = 20000
# With the value at false first it checks if there is already a model to load
FORCE_GENERATION_NEW_MODEL = False
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
        return output_t


def main():

    (TensorInputForTraining, TensorOutputForTraining,
     TensorInputForTest, passengerId) = generateTensorForTraining()

    model = getModel(TensorInputForTraining,
                     TensorOutputForTraining, nameModelFile='model.pth')

    res = generateOutputTest(TensorInputForTest, model)
    saveResults(res, passengerId)


def generateTensorForTraining():
    trainData: pd.DataFrame = loadData("trainEncoded.csv", "train.csv")
    testData: pd.DataFrame = loadData("testEncoded.csv", "test.csv")

    passengerId = testData.loc[:, "PassengerId"]

    resultTraining = trainData.loc[:, "Survived"]
    inputTraining = trainData.drop("Survived", axis=1)
    inputTraining = inputTraining.drop("PassengerId", axis=1)
    testData = testData.drop("PassengerId", axis=1)

    TensorInputForTraining = torch.from_numpy(
        inputTraining.to_numpy(dtype=np.float32, na_value=0))
    TensorResForTraining = torch.from_numpy(
        resultTraining.to_numpy(dtype=np.float32, na_value=0))
    TensorInputForTest = torch.from_numpy(
        testData.to_numpy(dtype=np.float32, na_value=0))

    normalizeTensor(TensorInputForTraining)
    normalizeTensor(TensorInputForTest)

    TensorResForTraining = TensorResForTraining.unsqueeze(1)

    return (TensorInputForTraining, TensorResForTraining, TensorInputForTest, passengerId)


def getModel(TensorInputForTraining, TensorOutputForTraining, nameModelFile):
    if os.path.exists(PATH_DATA+nameModelFile) and FORCE_GENERATION_NEW_MODEL == False:
        model = Model(TensorInputForTraining.shape[1])
        model.load_state_dict(torch.load(PATH_DATA+nameModelFile))
    else:
        model = generateModel(TensorInputForTraining, TensorOutputForTraining)
        torch.save(model.state_dict(), PATH_DATA+nameModelFile)

    return model


def generateOutputTest(TensorInputForTest, model):
    t_p_test: torch.Tensor = model(TensorInputForTest)
    res: torch.Tensor = setTo1Or0(t_p_test, 0.46)
    return res


def setTo1Or0(data_t, threshold=0.5):
    for i in range(data_t.shape[0]):
        if data_t[i] <= threshold:
            data_t[i] = 0
        else:
            data_t[i] = 1
    return data_t


def saveResults(res, passengerId):
    resDf = pd.DataFrame(res.detach().numpy())
    resDf = resDf.astype(int)

    resDf = resDf.set_axis(["Survived"], axis=1)
    resDf["PassengerId"] = passengerId
    resDf.to_csv(PATH_DATA+"results.csv", index=False,
                 columns=["PassengerId", "Survived"])


def loadData(namePdFile, nameFileData):
    if os.path.exists(PATH_DATA+namePdFile):
        data: pd.DataFrame = pd.read_csv(
            PATH_DATA+namePdFile, delimiter=',', header=0)
    else:
        data: pd.DataFrame = loadDataFromFile(nameFileData)
        data.to_csv(PATH_DATA+namePdFile, index=False)
    return data


def normalizeTensor(adultData):
    n_channels = adultData.shape[1]
    for i in range(0, n_channels):
        max = torch.max(adultData[:, i])
        if max > 1e-6:
            min = torch.min(adultData[:, i])
            adultData[:, i] = (adultData[:, i] - min) / max


def generateModel(TensorInputForTraining, TensorOutputForTraining):
    model = Model(TensorInputForTraining.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=LR)

    trainingLoop(n_epochs=N_EPOCHS, optimizer=optimizer, model=model, loss_fn=nn.MSELoss(
    ), TensorInputForTraining=TensorInputForTraining, TensorOutputForTraining=TensorOutputForTraining)

    return model


def loadDataFromFile(nameFile):
    df = pd.read_csv(PATH_DATA+nameFile, header=0, quotechar='"')
    df = df.drop("Name", axis=1)
    df = df.drop("Ticket", axis=1)
    df = df.drop("Fare", axis=1)
    df = df.drop("Cabin", axis=1)
    df = pd.get_dummies(df)

    return df


def trainingLoop(n_epochs, optimizer, model, loss_fn, TensorInputForTraining, TensorOutputForTraining):
    results = np.array([["Epoch", "Training Loss"]])
    for epoch in range(1, n_epochs+1):
        t_p_train = model(TensorInputForTraining)
        loss_train = loss_fn(t_p_train, TensorOutputForTraining)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        results = np.append(
            results, [[epoch, float(loss_train)]], axis=0)

        if epoch == 1 or epoch % 100 == 0:
            print("Epoch {}, Training loss {}".format(
                epoch, float(loss_train)))

    printGraphTrainingResults(results, n_epochs)


def printGraphTrainingResults(results, n_epochs):
    # n_epochs+1 because there are the names of the columns in the first row
    results.reshape(n_epochs+1, 2)
    index = results[1:, 0]
    data = results[1:, 1:]
    columns = results[0, 1:]
    resultDF = pd.DataFrame(
        data=data, index=index, columns=columns)
    resultDF['Training Loss'] = resultDF['Training Loss'].astype(float)
    resultDF.plot()
    plt.show()


main()