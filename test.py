import torch
import seaborn as sns
from model import GeneratorGAN, DiscriminatorGAN
import matplotlib.pyplot as plt

def testResults(dataLoader, totalEpochs, learningRate, kernelSize, paddingValue):
    genNet = GeneratorGAN(kernSize=5, padVal=2)
    discNet = DiscriminatorGAN(kernSize=5, padVal=2)

    fakeScores = []
    realScores = []

    genNet.load_state_dict(torch.load(f'G_{totalEpochs}_{learningRate}_{kernelSize}-{paddingValue}.pt'))
    discNet.load_state_dict(torch.load(f'D_{totalEpochs}_{learningRate}_{kernelSize}-{paddingValue}.pt'))

    for batchIndex, (batchImages, _) in enumerate(dataLoader):
        noiseVectors = torch.randn(128, 100, 1, 1)
        generatedImages = genNet(noiseVectors)
        fakePredictions = discNet(generatedImages).view(-1)
        fakeScores.extend(fakePredictions.tolist())
        realPredictions = discNet(batchImages).view(-1)
        realScores.extend(realPredictions.tolist())
    fakeResults = [1 if score >= 0.5 else 0 for score in fakeScores]
    realResults = [1 if score >= 0.5 else 0 for score in realScores]

    return fakeResults, realResults

def testPlotting(fakeOutputResults, realOutputResults):
    countMatrix = [[0, 0], [0, 0]]
    for result in fakeOutputResults:
        countMatrix[0][result] += 1
    for result in realOutputResults:
        countMatrix[1][result] += 1
    sns.heatmap(countMatrix, annot=True, fmt='d', cmap="Blues", xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title("Confusion Matrix of DCGAN Model After Being Trained on 200 Epochs")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
