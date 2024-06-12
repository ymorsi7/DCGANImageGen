import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np

def trainDiscriminator(discriminator, generator, optimizerD, realData, batchSize, device):
    lossFunction = nn.BCELoss()
    optimizerD.zero_grad()
    realData = realData.to(device)
    realLabels = torch.ones(realData.size(0), 1, device=device)
    realOutput = discriminator(realData)
    realLoss = lossFunction(realOutput, realLabels)
    realLoss.backward()

    noise = torch.randn(batchSize, 100, 1, 1, device=device)
    fakeData = generator(noise).detach()
    fakeLabels = torch.zeros(fakeData.size(0), 1, device=device)
    fakeOutput = discriminator(fakeData)
    fakeLoss = lossFunction(fakeOutput, fakeLabels)

    fakeLoss.backward()
    optimizerD.step()
    
    totalLoss = realLoss + fakeLoss
    return totalLoss

    return totalLoss

def trainGenerator(discriminator, generator, optimizerG, realData, batchSize, device):
    lossFunction = nn.BCELoss()
    generator.zero_grad()
    noise = torch.randn(batchSize, 100, 1, 1).to(device)
    output = discriminator(generator(noise))
    loss = lossFunction(output, torch.ones_like(output))
    loss.backward()

    optimizerG.step()
    return loss

def trainProcess(numEpochs, dataLoader, discriminator, generator, optimizerD, optimizerG, discLosses, genLosses, learningRate, kernelSize, padding, device):
    epochCounter = 0
    while epochCounter < numEpochs:
        for batchIndex, (realImages, _) in enumerate(dataLoader):
            discLossValue = trainDiscriminator(discriminator, generator, optimizerD, realImages, 128, device)
            genLossValue = trainGenerator(discriminator, generator, optimizerG, realImages, 128, device)
        discLosses.append(discLossValue.cpu().tolist())
        genLosses.append(genLossValue.cpu().tolist())
        epochCounter += 1

    torch.save(generator, f'Gen_{numEpochs}_{learningRate}_{kernelSize}-{padding}.pt')
    torch.save(discriminator, f'Disc_{numEpochs}_{learningRate}_{kernelSize}-{padding}.pt')
    print(f'After {numEpochs} epochs with learning rate = {learningRate}, kernel size = {kernelSize}, padding = {padding}: \tDiscriminator Loss: {discLossValue}\t Generator Loss: {genLossValue}\n')

    return discLosses, genLosses

def trainResults(numEpochs, generator, discLosses, genLosses, device):  
    fakeImages = generator(torch.randn(64, 100, 1, 1).to(device)).detach().cpu()
    fakeImages = vutils.make_grid(fakeImages, padding=True)
    plt.figure()
    plt.plot(discLosses, label="Discriminator")
    plt.plot(genLosses, label="Generator")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Discriminator and Generator Loss Over Epochs")

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(f"Fake Images After {numEpochs} Epochs")
    plt.imshow(np.transpose(fakeImages, (1, 2, 0)))
    plt.show()
